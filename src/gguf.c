// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */

/*
 * A GGUF reader, and the one matmul primitive the model is built out of.
 *
 * The file is mmap'd and never copied: a tensor is a pointer into the mapping
 * plus a shape. That matters on the board, where the model is a large part of
 * what RAM there is.
 */

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "charsiu_llm.h"

/* ---- half ---------------------------------------------------------------- */

static inline float half_to_float(uint16_t h)
{
	uint32_t s = (uint32_t)(h >> 15) << 31;
	uint32_t e = (h >> 10) & 0x1f;
	uint32_t m = h & 0x3ff;
	uint32_t bits;
	float f;

	if (e == 0) {
		if (m == 0) {
			bits = s;
		} else {
			/* subnormal: renormalise */
			e = 1;
			while (!(m & 0x400)) {
				m <<= 1;
				e--;
			}
			m &= 0x3ff;
			bits = s | ((e + 127 - 15) << 23) | (m << 13);
		}
	} else if (e == 0x1f) {
		bits = s | 0x7f800000u | (m << 13);
	} else {
		bits = s | ((e + 127 - 15) << 23) | (m << 13);
	}

	memcpy(&f, &bits, 4);
	return f;
}

/* ---- the type table ------------------------------------------------------ */

struct type_traits {
	const char *name;
	unsigned blck;
	unsigned size;
};

/* bf16 is f32's top half: put it back and the bottom sixteen bits are zero */
static inline float bf16_to_float(uint16_t h)
{
	union { uint32_t u; float f; } v;

	v.u = (uint32_t)h << 16;
	return v.f;
}

static const struct type_traits g_traits[] = {
	[GGML_F32]  = { "f32",  1,   4 },
	[GGML_F16]  = { "f16",  1,   2 },
	[GGML_Q4_0] = { "q4_0", 32, 18 },
	[GGML_Q4_1] = { "q4_1", 32, 20 },
	[GGML_Q8_0] = { "q8_0", 32, 34 },
	[GGML_Q6_K] = { "q6_K", 256, 210 },
	/*
	 * ⚠ bf16 IS f32 WITH THE BOTTOM SIXTEEN BITS THROWN AWAY, not a
	 * different float. gemma4's per_layer_model_proj arrives as this and
	 * nothing else in any file here has, which is why it is the only
	 * non-quantised type that had to be added rather than already being
	 * present: converting one is a shift, and reading it as f16 -- same
	 * width, different exponent -- would give numbers that are wrong by
	 * factors of 2^112 without anything failing.
	 */
	[GGML_BF16] = { "bf16", 1,   2 },
};

static const struct type_traits *traits_of(uint32_t t)
{
	if (t >= sizeof(g_traits) / sizeof(g_traits[0]) || !g_traits[t].name)
		return NULL;
	return &g_traits[t];
}

const char *ggml_type_name(uint32_t type)
{
	const struct type_traits *tt = traits_of(type);

	return tt ? tt->name : "unsupported";
}

/* ---- the cursor ---------------------------------------------------------- */

struct cur {
	const uint8_t *p;
	const uint8_t *end;
	int bad;
};

static int cur_take(struct cur *c, void *dst, size_t n)
{
	if (c->bad || (size_t)(c->end - c->p) < n) {
		c->bad = 1;
		return -1;
	}
	if (dst)
		memcpy(dst, c->p, n);
	c->p += n;
	return 0;
}

static uint64_t cur_u64(struct cur *c)
{
	uint64_t v = 0;

	cur_take(c, &v, 8);
	return v;
}

static uint32_t cur_u32(struct cur *c)
{
	uint32_t v = 0;

	cur_take(c, &v, 4);
	return v;
}

/* A GGUF string is a u64 length and that many bytes, not NUL terminated. */
static const char *cur_str(struct cur *c, uint64_t *len)
{
	uint64_t n = cur_u64(c);
	const char *s;

	if (c->bad || (uint64_t)(c->end - c->p) < n) {
		c->bad = 1;
		*len = 0;
		return NULL;
	}
	s = (const char *)c->p;
	c->p += n;
	*len = n;
	return s;
}

static size_t vtype_size(uint32_t t)
{
	switch (t) {
	case GGUF_V_U8: case GGUF_V_I8: case GGUF_V_BOOL: return 1;
	case GGUF_V_U16: case GGUF_V_I16: return 2;
	case GGUF_V_U32: case GGUF_V_I32: case GGUF_V_F32: return 4;
	case GGUF_V_U64: case GGUF_V_I64: case GGUF_V_F64: return 8;
	default: return 0;   /* string and array are not fixed width */
	}
}

static int read_value(struct cur *c, struct gguf_kv *kv, uint32_t type)
{
	kv->type = type;

	switch (type) {
	case GGUF_V_U8:   { uint8_t  v = 0; cur_take(c, &v, 1); kv->val.u = v; return 0; }
	case GGUF_V_I8:   { int8_t   v = 0; cur_take(c, &v, 1); kv->val.i = v; return 0; }
	case GGUF_V_U16:  { uint16_t v = 0; cur_take(c, &v, 2); kv->val.u = v; return 0; }
	case GGUF_V_I16:  { int16_t  v = 0; cur_take(c, &v, 2); kv->val.i = v; return 0; }
	case GGUF_V_U32:  { uint32_t v = 0; cur_take(c, &v, 4); kv->val.u = v; return 0; }
	case GGUF_V_I32:  { int32_t  v = 0; cur_take(c, &v, 4); kv->val.i = v; return 0; }
	case GGUF_V_F32:  { float    v = 0; cur_take(c, &v, 4); kv->val.f = v; return 0; }
	case GGUF_V_BOOL: { uint8_t  v = 0; cur_take(c, &v, 1); kv->val.u = v; return 0; }
	case GGUF_V_U64:  { kv->val.u = cur_u64(c); return 0; }
	case GGUF_V_I64:  { int64_t  v = 0; cur_take(c, &v, 8); kv->val.i = v; return 0; }
	case GGUF_V_F64:  { double   v = 0; cur_take(c, &v, 8); kv->val.f = v; return 0; }
	case GGUF_V_STRING:
		kv->str = cur_str(c, &kv->str_len);
		return c->bad ? -1 : 0;
	case GGUF_V_ARRAY: {
		uint32_t at = cur_u32(c);
		uint64_t n = cur_u64(c);
		size_t w = vtype_size(at);

		kv->arr_type = at;
		kv->arr_len = n;
		kv->arr = c->p;

		if (w) {
			if (n && n > (uint64_t)(c->end - c->p) / w) {
				c->bad = 1;
				return -1;
			}
			c->p += n * w;
		} else if (at == GGUF_V_STRING) {
			/* variable width: walk it so the cursor lands right */
			for (uint64_t i = 0; i < n && !c->bad; i++) {
				uint64_t l;

				cur_str(c, &l);
			}
		} else {
			c->bad = 1;   /* nested arrays are not in any file we read */
			return -1;
		}
		return c->bad ? -1 : 0;
	}
	default:
		c->bad = 1;
		return -1;
	}
}

/* ---- open ---------------------------------------------------------------- */

int gguf_open(struct gguf *g, const char *path)
{
	struct stat st;
	struct cur c;
	char magic[4] = { 0 };
	uint64_t off;

	memset(g, 0, sizeof(*g));
	g->fd = open(path, O_RDONLY);
	if (g->fd < 0) {
		fprintf(stderr, "gguf: open %s: %s\n", path, strerror(errno));
		return -1;
	}
	if (fstat(g->fd, &st) < 0 || st.st_size < 24) {
		fprintf(stderr, "gguf: %s is not a gguf file\n", path);
		close(g->fd);
		return -1;
	}
	g->map_size = (size_t)st.st_size;
	g->map = mmap(NULL, g->map_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
	if (g->map == MAP_FAILED) {
		fprintf(stderr, "gguf: mmap: %s\n", strerror(errno));
		close(g->fd);
		return -1;
	}

	c.p = g->map;
	c.end = g->map + g->map_size;
	c.bad = 0;

	cur_take(&c, magic, 4);
	if (memcmp(magic, "GGUF", 4)) {
		fprintf(stderr, "gguf: bad magic in %s\n", path);
		goto fail;
	}
	g->version = cur_u32(&c);
	if (g->version < 2 || g->version > 3) {
		fprintf(stderr, "gguf: version %u is not supported\n", g->version);
		goto fail;
	}
	g->n_tensors = cur_u64(&c);
	g->n_kv = cur_u64(&c);
	if (c.bad || g->n_tensors > 100000 || g->n_kv > 100000) {
		fprintf(stderr, "gguf: header out of range\n");
		goto fail;
	}

	g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof(*g->kv));
	g->t = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(*g->t));
	if (!g->kv || !g->t)
		goto fail;

	for (uint64_t i = 0; i < g->n_kv; i++) {
		uint64_t klen;
		const char *k = cur_str(&c, &klen);
		uint32_t vt;

		if (c.bad)
			goto fail;
		g->kv[i].key = malloc(klen + 1);
		if (!g->kv[i].key)
			goto fail;
		memcpy(g->kv[i].key, k, klen);
		g->kv[i].key[klen] = 0;

		vt = cur_u32(&c);
		if (read_value(&c, &g->kv[i], vt) < 0) {
			fprintf(stderr, "gguf: bad value for key %s\n", g->kv[i].key);
			goto fail;
		}
	}

	g->alignment = 32;
	{
		const struct gguf_kv *a = gguf_find(g, "general.alignment");

		if (a && a->type == GGUF_V_U32 && a->val.u)
			g->alignment = a->val.u;
	}

	for (uint64_t i = 0; i < g->n_tensors; i++) {
		struct gguf_tensor *t = &g->t[i];
		uint64_t nlen;
		const char *nm = cur_str(&c, &nlen);
		const struct type_traits *tt;
		uint64_t ne = 1;

		if (c.bad)
			goto fail;
		if (nlen >= sizeof(t->name))
			nlen = sizeof(t->name) - 1;
		memcpy(t->name, nm, nlen);
		t->name[nlen] = 0;

		t->n_dims = cur_u32(&c);
		if (t->n_dims > 4) {
			fprintf(stderr, "gguf: %s has %u dims\n", t->name, t->n_dims);
			goto fail;
		}
		t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
		for (unsigned d = 0; d < t->n_dims; d++)
			t->ne[d] = cur_u64(&c);
		t->type = cur_u32(&c);
		t->offset = cur_u64(&c);
		if (c.bad)
			goto fail;

		for (unsigned d = 0; d < 4; d++)
			ne *= t->ne[d];
		tt = traits_of(t->type);
		if (!tt) {
			/*
			 * Not fatal on its own: a file may hold a tensor this
			 * build cannot read and still be usable if the model
			 * never touches it. It fails later, by name.
			 */
			t->nbytes = 0;
			continue;
		}
		if (ne % tt->blck) {
			fprintf(stderr, "gguf: %s: %llu elements is not a multiple of the %s block\n",
				t->name, (unsigned long long)ne, tt->name);
			goto fail;
		}
		t->nbytes = ne / tt->blck * tt->size;
	}

	/* the tensor data begins at the next alignment boundary */
	off = (uint64_t)(c.p - g->map);
	off = (off + g->alignment - 1) / g->alignment * g->alignment;
	if (off > g->map_size) {
		fprintf(stderr, "gguf: data offset past the end of the file\n");
		goto fail;
	}
	g->data = g->map + off;

	for (uint64_t i = 0; i < g->n_tensors; i++) {
		struct gguf_tensor *t = &g->t[i];

		if (t->offset + t->nbytes > g->map_size - off) {
			fprintf(stderr, "gguf: %s runs past the end of the file\n", t->name);
			goto fail;
		}
		t->data = g->data + t->offset;
	}

	return 0;

fail:
	gguf_close(g);
	return -1;
}

void gguf_close(struct gguf *g)
{
	if (g->kv) {
		for (uint64_t i = 0; i < g->n_kv; i++)
			free(g->kv[i].key);
		free(g->kv);
	}
	free(g->t);
	if (g->map && g->map != MAP_FAILED)
		munmap((void *)g->map, g->map_size);
	if (g->fd >= 0)
		close(g->fd);
	memset(g, 0, sizeof(*g));
	g->fd = -1;
}

const struct gguf_kv *gguf_find(const struct gguf *g, const char *key)
{
	for (uint64_t i = 0; i < g->n_kv; i++)
		if (!strcmp(g->kv[i].key, key))
			return &g->kv[i];
	return NULL;
}

int gguf_get_u32(const struct gguf *g, const char *key, uint32_t *out)
{
	const struct gguf_kv *kv = gguf_find(g, key);

	if (!kv)
		return -1;
	switch (kv->type) {
	case GGUF_V_U8: case GGUF_V_U16: case GGUF_V_U32: case GGUF_V_U64:
	case GGUF_V_BOOL:
		*out = (uint32_t)kv->val.u;
		return 0;
	case GGUF_V_I8: case GGUF_V_I16: case GGUF_V_I32: case GGUF_V_I64:
		*out = (uint32_t)kv->val.i;
		return 0;
	default:
		return -1;
	}
}

int gguf_get_f32(const struct gguf *g, const char *key, float *out)
{
	const struct gguf_kv *kv = gguf_find(g, key);

	if (!kv)
		return -1;
	if (kv->type == GGUF_V_F32 || kv->type == GGUF_V_F64) {
		*out = (float)kv->val.f;
		return 0;
	}
	return -1;
}

int gguf_get_str(const struct gguf *g, const char *key, char *out, size_t max)
{
	const struct gguf_kv *kv = gguf_find(g, key);
	uint64_t n;

	if (!kv || kv->type != GGUF_V_STRING || !max)
		return -1;
	n = kv->str_len;
	if (n > max - 1)
		n = max - 1;
	memcpy(out, kv->str, n);
	out[n] = 0;
	return 0;
}

const struct gguf_tensor *gguf_tensor(const struct gguf *g, const char *name)
{
	for (uint64_t i = 0; i < g->n_tensors; i++)
		if (!strcmp(g->t[i].name, name))
			return &g->t[i];
	return NULL;
}

/* ---- the arithmetic ------------------------------------------------------ */

/*
 * Blocks, straight out of ggml. Kept here rather than included so this file has
 * no ggml dependency: the layouts are part of the file format the reader claims
 * to read, and a reader that cannot state them is not reading anything.
 */
struct block_q8_0 { uint16_t d; int8_t  qs[32]; };
struct block_q4_0 { uint16_t d; uint8_t qs[16]; };
struct block_q4_1 { uint16_t d; uint16_t m; uint8_t qs[16]; };
struct block_q6_K {
	uint8_t ql[128];
	uint8_t qh[64];
	int8_t  scales[16];
	uint16_t d;
};

/* Shared by both paths: q6_K is unpacked to f32 first either way. */
static void deq_q6_K(const struct block_q6_K *b, float *dst)
{
	float d = half_to_float(b->d);

	for (unsigned n = 0; n < 2; n++) {
		const uint8_t *ql = b->ql + n * 64;
		const uint8_t *qh = b->qh + n * 32;
		const int8_t *sc = b->scales + n * 8;
		float *y = dst + n * 128;

		for (unsigned l = 0; l < 32; l++) {
			int is = l / 16;
			int8_t q1 = (int8_t)((ql[l]      & 0xf) | (((qh[l] >> 0) & 3) << 4)) - 32;
			int8_t q2 = (int8_t)((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) - 32;
			int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
			int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

			y[l]      = d * sc[is + 0] * q1;
			y[l + 32] = d * sc[is + 2] * q2;
			y[l + 64] = d * sc[is + 4] * q3;
			y[l + 96] = d * sc[is + 6] * q4;
		}
	}
}

/*
 * The inner loops.
 *
 * The NEON versions are plain ARMv8.0: 128 bit vectors, half to single
 * conversion, and nothing from v8.2 -- no fp16 arithmetic and no SDOT. That is
 * deliberate. The RK3576 is a Cortex-A72 plus a Cortex-A53, both ARMv8.0, so a
 * kernel written against v8.2 would run on this development machine and not on
 * the board it is a baseline FOR.
 *
 * A slow CPU path would flatter the NPU. The comparison in step 3 is only
 * worth making against a CPU that is being asked to try.
 */
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
#include <arm_neon.h>

static float dot_f32(const float *w, const float *x, uint64_t n)
{
	float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
	uint64_t i = 0;
	float s;

	for (; i + 8 <= n; i += 8) {
		a0 = vfmaq_f32(a0, vld1q_f32(w + i), vld1q_f32(x + i));
		a1 = vfmaq_f32(a1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
	}
	s = vaddvq_f32(vaddq_f32(a0, a1));
	for (; i < n; i++)
		s += w[i] * x[i];
	return s;
}

static float dot_f16(const uint16_t *w, const float *x, uint64_t n)
{
	float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
	uint64_t i = 0;
	float s;

	for (; i + 8 <= n; i += 8) {
		uint16x8_t h = vld1q_u16(w + i);
		float32x4_t lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h)));
		float32x4_t hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h)));

		a0 = vfmaq_f32(a0, lo, vld1q_f32(x + i));
		a1 = vfmaq_f32(a1, hi, vld1q_f32(x + i + 4));
	}
	s = vaddvq_f32(vaddq_f32(a0, a1));
	for (; i < n; i++)
		s += half_to_float(w[i]) * x[i];
	return s;
}


/*
 * bf16, which is f32 with the bottom sixteen bits gone. Widening is a SHIFT,
 * not a conversion: interleave each half with a zero vector and the result is
 * already the float. There is no vcvt for it and none is needed.
 */
static float dot_bf16(const uint16_t *w, const float *x, uint64_t n)
{
	float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
	uint16x8_t z = vdupq_n_u16(0);
	uint64_t i = 0;
	float s;

	for (; i + 8 <= n; i += 8) {
		uint16x8_t h = vld1q_u16(w + i);
		float32x4_t lo = vreinterpretq_f32_u16(vzip1q_u16(z, h));
		float32x4_t hi = vreinterpretq_f32_u16(vzip2q_u16(z, h));

		a0 = vfmaq_f32(a0, lo, vld1q_f32(x + i));
		a1 = vfmaq_f32(a1, hi, vld1q_f32(x + i + 4));
	}
	s = vaddvq_f32(vaddq_f32(a0, a1));
	for (; i < n; i++)
		s += bf16_to_float(w[i]) * x[i];
	return s;
}

/*
 * Sixteen int8 against sixteen floats.
 *
 * A macro rather than a function taking float32x4_t acc[4]: an array parameter
 * is a pointer, so gcc spilled the accumulators to the stack and reloaded them
 * on every call. That cost q4_0 more than the nibble unpacking did -- it came
 * out at half of q8_0's tokens per second while reading half the bytes, which
 * is the wrong way round for a kernel that is supposed to be bandwidth bound.
 */
#define FMA_I8X16(a0, a1, a2, a3, q, xp) do {                                  \
	int16x8_t _l = vmovl_s8(vget_low_s8(q));                               \
	int16x8_t _h = vmovl_s8(vget_high_s8(q));                              \
	(a0) = vfmaq_f32((a0), vcvtq_f32_s32(vmovl_s16(vget_low_s16(_l))),  vld1q_f32((xp)));      \
	(a1) = vfmaq_f32((a1), vcvtq_f32_s32(vmovl_s16(vget_high_s16(_l))), vld1q_f32((xp) + 4));  \
	(a2) = vfmaq_f32((a2), vcvtq_f32_s32(vmovl_s16(vget_low_s16(_h))),  vld1q_f32((xp) + 8));  \
	(a3) = vfmaq_f32((a3), vcvtq_f32_s32(vmovl_s16(vget_high_s16(_h))), vld1q_f32((xp) + 12)); \
} while (0)

#define HSUM4(a0, a1, a2, a3) \
	vaddvq_f32(vaddq_f32(vaddq_f32((a0), (a1)), vaddq_f32((a2), (a3))))

static float dot_q8_0(const struct block_q8_0 *b, const float *x, uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
		float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);

		FMA_I8X16(a0, a1, a2, a3, vld1q_s8(b[i].qs),      x + i * 32);
		FMA_I8X16(a0, a1, a2, a3, vld1q_s8(b[i].qs + 16), x + i * 32 + 16);

		s += half_to_float(b[i].d) * HSUM4(a0, a1, a2, a3);
	}
	return s;
}

static float dot_q4_0(const struct block_q4_0 *b, const float *x, uint64_t nb)
{
	const int8x16_t bias = vdupq_n_s8(8);
	const uint8x16_t mask = vdupq_n_u8(0x0f);
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
		float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
		uint8x16_t p = vld1q_u8(b[i].qs);
		int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p, mask)), bias);
		int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p, 4)), bias);

		/* the low nibbles are elements 0..15, the high ones 16..31 */
		FMA_I8X16(a0, a1, a2, a3, lo, x + i * 32);
		FMA_I8X16(a0, a1, a2, a3, hi, x + i * 32 + 16);

		s += half_to_float(b[i].d) * HSUM4(a0, a1, a2, a3);
	}
	return s;
}

static float dot_q4_1(const struct block_q4_1 *b, const float *x, uint64_t nb)
{
	const uint8x16_t mask = vdupq_n_u8(0x0f);
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
		float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
		float32x4_t x0 = vdupq_n_f32(0), x1 = vdupq_n_f32(0);
		uint8x16_t p = vld1q_u8(b[i].qs);
		const float *xp = x + i * 32;

		/* q4_1 nibbles are unsigned and the offset is a separate term */
		FMA_I8X16(a0, a1, a2, a3,
			  vreinterpretq_s8_u8(vandq_u8(p, mask)), xp);
		FMA_I8X16(a0, a1, a2, a3,
			  vreinterpretq_s8_u8(vshrq_n_u8(p, 4)), xp + 16);

		for (unsigned j = 0; j < 32; j += 8) {
			x0 = vaddq_f32(x0, vld1q_f32(xp + j));
			x1 = vaddq_f32(x1, vld1q_f32(xp + j + 4));
		}

		s += half_to_float(b[i].d) * HSUM4(a0, a1, a2, a3) +
		     half_to_float(b[i].m) * vaddvq_f32(vaddq_f32(x0, x1));
	}
	return s;
}

/* one int8x16 against sixteen floats, summed */
static inline float dot_i8x16_f32(int8x16_t q, const float *x)
{
	int16x8_t l = vmovl_s8(vget_low_s8(q));
	int16x8_t h = vmovl_s8(vget_high_s8(q));
	float32x4_t a0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))),  vld1q_f32(x));
	float32x4_t a1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))), vld1q_f32(x + 4));
	float32x4_t a2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),  vld1q_f32(x + 8));
	float32x4_t a3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))), vld1q_f32(x + 12));

	return HSUM4(a0, a1, a2, a3);
}

/*
 * q6_K matters more than its share of the tensor count suggests: llama.cpp
 * quantises token_embd (and so the tied output head) to q6_K even in a q4_0
 * file, and at a 128256 vocabulary that one tensor is a fifth of the weights
 * and the single biggest matmul in a decode step. Leaving it scalar made a
 * q4_0 model run at HALF a q8_0 model's tokens per second while reading half
 * the bytes -- backwards for a kernel that should be bandwidth bound, and the
 * reason to look at the file's actual types rather than its name.
 *
 * The dot is folded INTO the unpacking rather than run over a dequantised
 * buffer: a q6_K scale covers sixteen weights, so sixteen is also the group
 * this sums over, and the scale multiplies one number at the end of it.
 */
static float dot_q6_K(const struct block_q6_K *b, const float *x, uint64_t nb)
{
	const uint8x16_t m4 = vdupq_n_u8(0x0f);
	const uint8x16_t m3 = vdupq_n_u8(0x03);
	const int8x16_t bias = vdupq_n_s8(32);
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float d = half_to_float(b[i].d);

		for (unsigned n = 0; n < 2; n++) {
			const uint8_t *ql = b[i].ql + n * 64;
			const uint8_t *qh = b[i].qh + n * 32;
			const int8_t *sc = b[i].scales + n * 8;
			const float *y = x + i * 256 + n * 128;

			/* g = 0 is l 0..15 and g = 1 is l 16..31, one scale each */
			for (unsigned g = 0; g < 2; g++) {
				unsigned l0 = g * 16;
				uint8x16_t a = vld1q_u8(ql + l0);
				uint8x16_t c = vld1q_u8(ql + 32 + l0);
				uint8x16_t hb = vld1q_u8(qh + l0);
				int8x16_t q1, q2, q3, q4;

				q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vandq_u8(a, m4),
					vshlq_n_u8(vandq_u8(hb, m3), 4))), bias);
				q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vandq_u8(c, m4),
					vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 2), m3), 4))), bias);
				q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vshrq_n_u8(a, 4),
					vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 4), m3), 4))), bias);
				q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vshrq_n_u8(c, 4),
					vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 6), m3), 4))), bias);

				s += d * (float)sc[g]     * dot_i8x16_f32(q1, y + l0);
				s += d * (float)sc[g + 2] * dot_i8x16_f32(q2, y + 32 + l0);
				s += d * (float)sc[g + 4] * dot_i8x16_f32(q3, y + 64 + l0);
				s += d * (float)sc[g + 6] * dot_i8x16_f32(q4, y + 96 + l0);
			}
		}
	}
	return s;
}

#else /* the portable versions, which are also what the NEON ones are read against */

static float dot_f32(const float *w, const float *x, uint64_t n)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < n; i++)
		s += w[i] * x[i];
	return s;
}

static float dot_f16(const uint16_t *w, const float *x, uint64_t n)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < n; i++)
		s += half_to_float(w[i]) * x[i];
	return s;
}

static float dot_bf16(const uint16_t *w, const float *x, uint64_t n)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < n; i++)
		s += bf16_to_float(w[i]) * x[i];
	return s;
}

static float dot_q8_0(const struct block_q8_0 *b, const float *x, uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float d = half_to_float(b[i].d);
		float a = 0.0f;

		for (unsigned j = 0; j < 32; j++)
			a += (float)b[i].qs[j] * x[i * 32 + j];
		s += d * a;
	}
	return s;
}

static float dot_q4_0(const struct block_q4_0 *b, const float *x, uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float d = half_to_float(b[i].d);
		float a = 0.0f;

		/* nibble j feeds element j, its high half feeds element j + 16 */
		for (unsigned j = 0; j < 16; j++) {
			int lo = (b[i].qs[j] & 0x0f) - 8;
			int hi = (b[i].qs[j] >> 4) - 8;

			a += (float)lo * x[i * 32 + j];
			a += (float)hi * x[i * 32 + j + 16];
		}
		s += d * a;
	}
	return s;
}

static float dot_q4_1(const struct block_q4_1 *b, const float *x, uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float d = half_to_float(b[i].d);
		float m = half_to_float(b[i].m);
		float a = 0.0f, xs = 0.0f;

		for (unsigned j = 0; j < 16; j++) {
			a += (float)(b[i].qs[j] & 0x0f) * x[i * 32 + j];
			a += (float)(b[i].qs[j] >> 4) * x[i * 32 + j + 16];
			xs += x[i * 32 + j] + x[i * 32 + j + 16];
		}
		s += d * a + m * xs;
	}
	return s;
}

static float dot_q6_K(const struct block_q6_K *b, const float *x, uint64_t nb)
{
	float s = 0.0f;
	float tmp[256];

	/*
	 * A plain loop rather than dot_f32(): inlining a vector kernel over a
	 * 256 float array on the stack makes gcc warn about the residual
	 * iteration it can see is unreachable. q6_K is not a hot path here.
	 */
	for (uint64_t i = 0; i < nb; i++) {
		deq_q6_K(&b[i], tmp);
		for (unsigned j = 0; j < 256; j++)
			s += tmp[j] * x[i * 256 + j];
	}
	return s;
}

#endif

/* ---- the three kernels an attention is made of ---------------------------- */

/*
 * ⚠ THESE ARE FOR THE ATTENTION, WHICH IS NOT A MATMUL AGAINST A WEIGHT and so
 * has none of the machinery above. On the board it is 68% of a transcription
 * with every core already working on it: 1500 queries against 1500 keys, and
 * the innermost thing is a dot product of head_dim floats and a scaled add of
 * head_dim floats. Both were written as plain loops.
 *
 * ⚠ THE SUMMATION ORDER CHANGES. dot_f32 accumulates in eight lanes, so a
 * result is not bit identical to the scalar loop -- which is why the transcript
 * and the numpy cross check are run against this rather than assumed.
 */
float charsiu_dot_f32(const float *a, const float *b, uint64_t n)
{
	return dot_f32(a, b, n);
}

/*
 * ⚠⚠ AND THE THIRD ONE IS THE EXPONENTIAL, WHICH NOBODY COUNTED.
 *
 * A softmax over every patch against every patch asks for n^2 exponentials a
 * head. The vision tower is 1024 against 1024, twelve heads, twelve layers:
 * 151 MILLION of them for one picture. Round 367 measured glibc's expf on the
 * board at 23 ns an element, which puts 3.5 s under a stage the board reports
 * as 4.0 s -- so the arithmetic nobody had counted was most of a stage everyone
 * was reading as memory traffic.
 *
 * This does the whole of what a softmax's middle pass wants -- e^(x - m) in
 * place, and the sum back -- because splitting it into an exp pass and a sum
 * pass reads the row twice.
 *
 * ⚠ THE POLYNOMIAL IS THE SLIGHTLY WORSE OF THE TWO. glibc's expf is correctly
 * rounded and this is about one last bit out, the same trade llama.c's vexpq
 * makes for SiLU. There it is opt in because a near tie in greedy decoding
 * changes a word and the project's anchor sentence with it; here the consumer
 * is an embedding that goes through a projection and twelve more layers, and
 * the tower's own cross check has 2e-4 of room against a numpy reference that
 * uses neither. CHARSIU_EXACT_SOFTMAX is the control.
 */
static int sm_exact = -1;

static int softmax_exact(void)
{
	if (sm_exact < 0)
		sm_exact = getenv("CHARSIU_EXACT_SOFTMAX") != NULL;
	return sm_exact;
}

/*
 * ⚠ SETTABLE FOR THE SAME REASON THE SCHEDULE IS. Two runs of this host, one
 * with the polynomial and one without, disagreed with each other by more than
 * the thing being measured; the only reading worth having interleaves them in
 * one process.
 */
void charsiu_softmax_exact_set(int on)
{
	sm_exact = on ? 1 : 0;
}

float charsiu_expsum_f32(float *x, uint64_t n, float m)
{
	uint64_t i = 0;
	float s = 0.0f;

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	if (!softmax_exact()) {
		float32x4_t vm = vdupq_n_f32(m), acc = vdupq_n_f32(0.0f);

		for (; i + 4 <= n; i += 4) {
			float32x4_t e = charsiu_vexpq(vsubq_f32(vld1q_f32(x + i),
								vm));

			vst1q_f32(x + i, e);
			acc = vaddq_f32(acc, e);
		}
		s = vaddvq_f32(acc);
	}
#endif
	for (; i < n; i++) {
		x[i] = expf(x[i] - m);
		s += x[i];
	}
	return s;
}

/*
 * ⚠⚠ AND THE FOURTH IS THE ONE THE SCALED ADD SHOULD HAVE BEEN ALL ALONG.
 *
 * acc[u] += sum over the key tile of s[u][j] * v[j], for a BLOCK of queries at
 * once. Called one query and one key at a time -- which is what an axpy per
 * (query, key) is -- every FMA costs two vector loads and a vector store,
 * because the accumulator is fetched from memory and put back for each single
 * key. Three memory operations per FMA is not an attention, it is a memcpy with
 * arithmetic in it.
 *
 * Holding four accumulators in registers across the whole tile and reusing each
 * value row for all four of them makes it four vector loads and four scalar
 * loads for sixteen FMAs: half a memory operation per FMA, six times fewer.
 *
 * ⚠ AND IT IS STILL BIT IDENTICAL. For a given output element the sum over j
 * is in the same order, starting from the same accumulator; only the order the
 * ELEMENTS are visited in changes. The bench diffs it element wise against the
 * unblocked form rather than taking that on trust.
 *
 * Everything takes a stride: the values arrive as [n][n_embd] with the heads
 * side by side, so one head's row is head_dim floats every n_embd, and the
 * accumulator is either a contiguous block the fused kernel owns or a slice of
 * the output rows the three pass kernel writes in place.
 */
/*
 * ⚠ THE CONTROL, AND IT IS NOT OPTIONAL. This kernel is meant to be BIT
 * identical to the axpy per (query, key) it replaces, which means a stopwatch
 * is the only thing that can tell them apart -- and on a loaded host two builds
 * timed one after the other cannot. CHARSIU_PLAIN_PV puts the old form back so
 * the two can be interleaved in one process.
 */
static int pv_plain = -1;

void charsiu_pv_plain_set(int on)
{
	pv_plain = on ? 1 : 0;
}

static int pv_is_plain(void)
{
	if (pv_plain < 0)
		pv_plain = getenv("CHARSIU_PLAIN_PV") != NULL;
	return pv_plain;
}

void charsiu_pv_f32(float *acc, uint64_t astride, const float *v,
		    uint64_t vstride, const float *s, uint64_t sstride,
		    unsigned nq, unsigned nk, unsigned hd)
{
	unsigned u, j;
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	unsigned e0, u0;

	if (pv_is_plain()) {
		for (u = 0; u < nq; u++)
			for (j = 0; j < nk; j++)
				charsiu_axpy_f32(acc + (uint64_t)u * astride,
						 v + (uint64_t)j * vstride,
						 s[(uint64_t)u * sstride + j],
						 hd);
		return;
	}
	/*
	 * ⚠⚠ WRITTEN OUT, NOT LOOPED, AND THE FIRST VERSION WAS SLOWER THAN THE
	 * AXPY IT REPLACED. With the sixteen accumulators in a float32x4_t[4]
	 * and a `for (i = 0; i < 4; i++)` over them, -O2 does not unroll -- so
	 * they stayed in memory and the inner loop came out as four FMAs
	 * against four RELOADS of the accumulator, which is the exact cost this
	 * kernel exists to remove. It measured 0.94x. Sixteen named variables
	 * is what keeps them in registers.
	 */
#define PV_FMA(b0, b1, b2, b3, cs) do { \
	b0 = vfmaq_n_f32(b0, w0, cs); b1 = vfmaq_n_f32(b1, w1, cs); \
	b2 = vfmaq_n_f32(b2, w2, cs); b3 = vfmaq_n_f32(b3, w3, cs); \
} while (0)

	for (u0 = 0; u0 + 4 <= nq; u0 += 4) {
		const float *s0 = s + (uint64_t)u0 * sstride;
		const float *s1 = s0 + sstride, *s2 = s1 + sstride;
		const float *s3 = s2 + sstride;

		for (e0 = 0; e0 + 16 <= hd; e0 += 16) {
			float *a0 = acc + (uint64_t)u0 * astride + e0;
			float *a1 = a0 + astride, *a2 = a1 + astride;
			float *a3 = a2 + astride;
			float32x4_t p0 = vld1q_f32(a0), p1 = vld1q_f32(a0 + 4);
			float32x4_t p2 = vld1q_f32(a0 + 8);
			float32x4_t p3 = vld1q_f32(a0 + 12);
			float32x4_t q0 = vld1q_f32(a1), q1 = vld1q_f32(a1 + 4);
			float32x4_t q2 = vld1q_f32(a1 + 8);
			float32x4_t q3 = vld1q_f32(a1 + 12);
			float32x4_t t0 = vld1q_f32(a2), t1 = vld1q_f32(a2 + 4);
			float32x4_t t2 = vld1q_f32(a2 + 8);
			float32x4_t t3 = vld1q_f32(a2 + 12);
			float32x4_t z0 = vld1q_f32(a3), z1 = vld1q_f32(a3 + 4);
			float32x4_t z2 = vld1q_f32(a3 + 8);
			float32x4_t z3 = vld1q_f32(a3 + 12);

			for (j = 0; j < nk; j++) {
				const float *vj = v + (uint64_t)j * vstride + e0;
				float32x4_t w0 = vld1q_f32(vj);
				float32x4_t w1 = vld1q_f32(vj + 4);
				float32x4_t w2 = vld1q_f32(vj + 8);
				float32x4_t w3 = vld1q_f32(vj + 12);

				PV_FMA(p0, p1, p2, p3, s0[j]);
				PV_FMA(q0, q1, q2, q3, s1[j]);
				PV_FMA(t0, t1, t2, t3, s2[j]);
				PV_FMA(z0, z1, z2, z3, s3[j]);
			}
			vst1q_f32(a0, p0); vst1q_f32(a0 + 4, p1);
			vst1q_f32(a0 + 8, p2); vst1q_f32(a0 + 12, p3);
			vst1q_f32(a1, q0); vst1q_f32(a1 + 4, q1);
			vst1q_f32(a1 + 8, q2); vst1q_f32(a1 + 12, q3);
			vst1q_f32(a2, t0); vst1q_f32(a2 + 4, t1);
			vst1q_f32(a2 + 8, t2); vst1q_f32(a2 + 12, t3);
			vst1q_f32(a3, z0); vst1q_f32(a3 + 4, z1);
			vst1q_f32(a3 + 8, z2); vst1q_f32(a3 + 12, z3);
		}
		/*
		 * ⚠ THE TAIL IS THE OLD KERNEL, not a second copy of the new
		 * one. head_dim is 64 or 80 in every tower this reads, so this
		 * runs for the last 0 or 16 lanes and never for the hot part.
		 */
		if (e0 < hd)
			for (j = 0; j < nk; j++)
				for (u = u0; u < u0 + 4; u++)
					charsiu_axpy_f32(acc + (uint64_t)u *
							 astride + e0,
							 v + (uint64_t)j *
							 vstride + e0,
							 s[(uint64_t)u *
							   sstride + j],
							 hd - e0);
	}
	for (u = u0; u < nq; u++)
		for (j = 0; j < nk; j++)
			charsiu_axpy_f32(acc + (uint64_t)u * astride,
					 v + (uint64_t)j * vstride,
					 s[(uint64_t)u * sstride + j], hd);
#undef PV_FMA
#else
	for (u = 0; u < nq; u++)
		for (j = 0; j < nk; j++)
			charsiu_axpy_f32(acc + (uint64_t)u * astride,
					 v + (uint64_t)j * vstride,
					 s[(uint64_t)u * sstride + j], hd);
#endif
}

void charsiu_axpy_f32(float *y, const float *x, float a, uint64_t n)
{
#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)
	float32x4_t va = vdupq_n_f32(a);
	uint64_t i = 0;

	for (; i + 8 <= n; i += 8) {
		vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i),
					   vld1q_f32(x + i), va));
		vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4),
					       vld1q_f32(x + i + 4), va));
	}
	for (; i < n; i++)
		y[i] += a * x[i];
#else
	uint64_t i;

	for (i = 0; i < n; i++)
		y[i] += a * x[i];
#endif
}

/* ---- the activation, quantised once per matvec --------------------------- */

/*
 * Converting every WEIGHT to a float to multiply it by a float activation is
 * the expensive way round: there are N*K weights and only K activations. Quantise
 * the activation once instead and the dot product becomes integer, with one
 * multiply per 32 element block to put it back on scale.
 *
 * This is what makes llama.cpp roughly two and a half times faster than the
 * f32 path here, and it is also the operand shape the NPU wants, so the CPU
 * fallback and the NPU job can be fed from the same buffer.
 *
 * It is an APPROXIMATION and the exact path is kept: CHARSIU_NO_QACT restores
 * it, and that is the control.
 */
int charsiu_act_alloc(struct charsiu_act *a, int max_n)
{
	int nb = (max_n + 31) / 32;

	memset(a, 0, sizeof(*a));
	a->q = calloc((size_t)nb * 32, 1);
	a->d = calloc((size_t)nb, sizeof(float));
	a->bs = calloc((size_t)nb, sizeof(float));
	a->q1 = calloc((size_t)nb * 32, 1);
	if (!a->q || !a->d || !a->bs || !a->q1) {
		charsiu_act_free(a);
		return -1;
	}
	return 0;
}

void charsiu_act_free(struct charsiu_act *a)
{
	free(a->q);
	free(a->d);
	free(a->bs);
	free(a->q1);
	memset(a, 0, sizeof(*a));
}

/*
 * LAZY SINCE ROUND 377, and round 376 is why: this cost 8.09 ms a token on
 * the board, 11 percent of a 64 token one, and the int4 path reads the FLOAT
 * activation and looks at neither result.
 *
 * Two quantisations happen here. The blocks of 32 are what gguf_matvec dot
 * kernels take; the single scale q1 is the shape the NPU int8 operand wants
 * and is what npu_matvec falls back on. In a fully routed int4 run NOTHING
 * reads either, and they were computed 65 times a token anyway: 231 thousand
 * elements, twice over.
 *
 * So this records the vector and the consumers ask. The realise calls live in
 * llama.c BEFORE the pool fans out, never inside gguf_matvec or npu_matvec --
 * those run on four worker threads and filling a shared buffer from all of
 * them would be a race.
 *
 * CHARSIU_ACT_EAGER does both up front, which is every round before 377.
 */
void charsiu_act_blocks(struct charsiu_act *a)
{
	const float *x = a->f;
	int n = a->n;

	if (a->quantised || !a->q)
		return;
	for (int i = 0; i < a->nb; i++) {
		int base = i * 32;
		float amax = 0.0f, sum = 0.0f, d, id;

		for (int j = 0; j < 32; j++) {
			float v = base + j < n ? x[base + j] : 0.0f;
			float av = v < 0.0f ? -v : v;

			if (av > amax)
				amax = av;
			sum += v;
		}
		d = amax / 127.0f;
		id = d != 0.0f ? 1.0f / d : 0.0f;
		a->d[i] = d;
		a->bs[i] = sum;
		for (int j = 0; j < 32; j++) {
			float v = base + j < n ? x[base + j] : 0.0f;

			/*
			 * lrintf, not roundf. ggml's REFERENCE C uses roundf
			 * (ties away from zero) but its ARM path quantises with
			 * vcvtnq_s32_f32, which is ties to EVEN, and the ARM
			 * path is the one that runs. Matching roundf here made
			 * the worst logit disagreement with it twice as large
			 * and cost 9% of the tokens per second.
			 */
			a->q[base + j] = (int8_t)lrintf(v * id);
		}
	}
	a->quantised = 1;
}

void charsiu_act_q1(struct charsiu_act *a)
{
	const float *x = a->f;
	int n = a->n;

	if (a->q1_valid || !a->npu_ok)
		return;
	float amax = 0.0f, d, id;

	for (int i = 0; i < n; i++) {
		float av = x[i] < 0.0f ? -x[i] : x[i];

		if (av > amax)
			amax = av;
	}
	d = amax / 127.0f;
	id = d != 0.0f ? 1.0f / d : 0.0f;
	a->d1 = d;
	for (int i = 0; i < a->nb * 32; i++) {
		int v = i < n ? (int)lrintf(x[i] * id) : 0;

		if (v > 127) v = 127;
		if (v < -127) v = -127;
		a->q1[i] = (int8_t)v;
	}
	a->q1_valid = 1;
}

void charsiu_act_set(struct charsiu_act *a, const float *x, int n)
{
	static int off = -1, npu = -1, eager = -1;

	if (off < 0) {
		const char *e = getenv("CHARSIU_NO_QACT");

		off = e && *e != '0';
	}
	if (npu < 0) {
		const char *e = getenv("CHARSIU_NPU_QUANT");

		npu = e && *e != '0';
	}
	if (eager < 0)
		eager = getenv("CHARSIU_ACT_EAGER") != NULL;

	a->f = x;
	a->n = n;
	a->nb = (n + 31) / 32;
	a->quantised = 0;
	a->q1_valid = 0;
	a->npu_ok = npu && !off && a->q1 != NULL;
	if (off || !a->q)
		return;
	if (eager) {
		charsiu_act_blocks(a);
		charsiu_act_q1(a);
	}
}

/* ---- integer dot products ------------------------------------------------ */

#if defined(__ARM_NEON) && !defined(CHARSIU_NO_NEON)

/* No SDOT on an ARMv8.0 core, so widen and pairwise accumulate. int8 by int8
 * is at most 16129, which an int16 lane holds, and 32 of those an int32. */
static inline int32x4_t iacc16(int32x4_t acc, int8x16_t w, int8x16_t x)
{
	acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w),  vget_low_s8(x)));
	acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w), vget_high_s8(x)));
	return acc;
}

static inline int32_t idot16(int8x16_t w, const int8_t *x)
{
	return vaddvq_s32(iacc16(vdupq_n_s32(0), w, vld1q_s8(x)));
}

static inline int32_t idot32(const int8_t *w, const int8_t *x)
{
	int32x4_t acc = vdupq_n_s32(0);

	acc = iacc16(acc, vld1q_s8(w), vld1q_s8(x));
	acc = iacc16(acc, vld1q_s8(w + 16), vld1q_s8(x + 16));
	return vaddvq_s32(acc);
}

static inline void q4_0_nibbles(const uint8_t *qs, int8x16_t *lo, int8x16_t *hi)
{
	const uint8x16_t m4 = vdupq_n_u8(0x0f);
	const int8x16_t eight = vdupq_n_s8(8);
	uint8x16_t p = vld1q_u8(qs);

	*lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p, m4)), eight);
	*hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p, 4)), eight);
}

static float dotq_q4_0(const struct block_q4_0 *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		int8x16_t lo, hi;
		int32x4_t acc = vdupq_n_s32(0);

		q4_0_nibbles(b[i].qs, &lo, &hi);
		acc = iacc16(acc, lo, vld1q_s8(a->q + i * 32));
		acc = iacc16(acc, hi, vld1q_s8(a->q + i * 32 + 16));
		s += half_to_float(b[i].d) * a->d[i] * (float)vaddvq_s32(acc);
	}
	return s;
}

/*
 * ⚠ THE WEIGHT ROW IS THE EXPENSIVE PART, AND PREFILL WAS PAYING FOR IT ONCE A
 * TOKEN. Measured: prompt throughput was FLAT at ~64-68 tok/s whatever the
 * length, which is the signature of no batching -- 241 prompt tokens read the
 * whole weight set 241 times. These take M activations against one row, so the
 * row is unpacked and, more to the point, LOADED FROM MEMORY once for all M.
 *
 * The arithmetic is unchanged: each output is the same sum in the same order
 * as the single-vector kernel, so a batch of one is bit-identical to it.
 */
static void dotq_q4_0_m(const struct block_q4_0 *b, const struct charsiu_act *a,
			unsigned m, uint64_t nb, float *out, uint64_t ostride)
{
	float s[CHARSIU_BATCH_MAX];
	unsigned j;

	for (j = 0; j < m; j++)
		s[j] = 0.0f;
	for (uint64_t i = 0; i < nb; i++) {
		int8x16_t lo, hi;
		float d;

		q4_0_nibbles(b[i].qs, &lo, &hi);
		d = half_to_float(b[i].d);
		for (j = 0; j < m; j++) {
			int32x4_t acc = vdupq_n_s32(0);

			acc = iacc16(acc, lo, vld1q_s8(a[j].q + i * 32));
			acc = iacc16(acc, hi, vld1q_s8(a[j].q + i * 32 + 16));
			s[j] += d * a[j].d[i] * (float)vaddvq_s32(acc);
		}
	}
	for (j = 0; j < m; j++)
		out[j * ostride] = s[j];
}

static void dotq_q8_0_m(const struct block_q8_0 *b, const struct charsiu_act *a,
			unsigned m, uint64_t nb, float *out, uint64_t ostride)
{
	float s[CHARSIU_BATCH_MAX];
	unsigned j;

	for (j = 0; j < m; j++)
		s[j] = 0.0f;
	for (uint64_t i = 0; i < nb; i++) {
		int8x16_t w0 = vld1q_s8(b[i].qs);
		int8x16_t w1 = vld1q_s8(b[i].qs + 16);
		float d = half_to_float(b[i].d);

		for (j = 0; j < m; j++) {
			int32x4_t acc = vdupq_n_s32(0);

			acc = iacc16(acc, w0, vld1q_s8(a[j].q + i * 32));
			acc = iacc16(acc, w1, vld1q_s8(a[j].q + i * 32 + 16));
			s[j] += d * a[j].d[i] * (float)vaddvq_s32(acc);
		}
	}
	for (j = 0; j < m; j++)
		out[j * ostride] = s[j];
}

static float dotq_q4_1(const struct block_q4_1 *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	const uint8x16_t m4 = vdupq_n_u8(0x0f);
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		uint8x16_t p = vld1q_u8(b[i].qs);
		int32x4_t acc = vdupq_n_s32(0);

		acc = iacc16(acc, vreinterpretq_s8_u8(vandq_u8(p, m4)),
			     vld1q_s8(a->q + i * 32));
		acc = iacc16(acc, vreinterpretq_s8_u8(vshrq_n_u8(p, 4)),
			     vld1q_s8(a->q + i * 32 + 16));
		s += half_to_float(b[i].d) * a->d[i] * (float)vaddvq_s32(acc) +
		     half_to_float(b[i].m) * a->bs[i];
	}
	return s;
}

static float dotq_q8_0(const struct block_q8_0 *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++)
		s += half_to_float(b[i].d) * a->d[i] *
		     (float)idot32(b[i].qs, a->q + i * 32);
	return s;
}

/*
 * A q6_K scale covers sixteen weights and an activation block covers
 * thirty-two, so a sixteen run is always one half of one activation block and
 * the two scales multiply out to a single float at the end of it.
 */
static float dotq_q6_K(const struct block_q6_K *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	const uint8x16_t m4 = vdupq_n_u8(0x0f);
	const uint8x16_t m3 = vdupq_n_u8(0x03);
	const int8x16_t bias = vdupq_n_s8(32);
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float d = half_to_float(b[i].d);

		for (unsigned n = 0; n < 2; n++) {
			const uint8_t *ql = b[i].ql + n * 64;
			const uint8_t *qh = b[i].qh + n * 32;
			const int8_t *sc = b[i].scales + n * 8;
			const int8_t *xq = a->q + i * 256 + n * 128;
			const float *xd = a->d + i * 8 + n * 4;

			for (unsigned g = 0; g < 2; g++) {
				unsigned l0 = g * 16;
				uint8x16_t p0 = vld1q_u8(ql + l0);
				uint8x16_t p1 = vld1q_u8(ql + 32 + l0);
				uint8x16_t hb = vld1q_u8(qh + l0);
				int8x16_t q[4];

				q[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vandq_u8(p0, m4),
					vshlq_n_u8(vandq_u8(hb, m3), 4))), bias);
				q[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vandq_u8(p1, m4),
					vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 2), m3), 4))), bias);
				q[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vshrq_n_u8(p0, 4),
					vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 4), m3), 4))), bias);
				q[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
					vshrq_n_u8(p1, 4),
					vshlq_n_u8(vandq_u8(vshrq_n_u8(hb, 6), m3), 4))), bias);

				for (unsigned t = 0; t < 4; t++)
					s += d * (float)sc[g + 2 * t] * xd[t] *
					     (float)idot16(q[t], xq + t * 32 + l0);
			}
		}
	}
	return s;
}

#else /* the portable integer kernels */

static inline int32_t idot_n(const int8_t *w, const int8_t *x, int n)
{
	int32_t t = 0;

	for (int j = 0; j < n; j++)
		t += (int32_t)w[j] * (int32_t)x[j];
	return t;
}

static float dotq_q4_0(const struct block_q4_0 *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		const int8_t *x = a->q + i * 32;
		int32_t t = 0;

		for (unsigned j = 0; j < 16; j++) {
			t += (int32_t)((b[i].qs[j] & 0x0f) - 8) * x[j];
			t += (int32_t)((b[i].qs[j] >> 4) - 8) * x[j + 16];
		}
		s += half_to_float(b[i].d) * a->d[i] * (float)t;
	}
	return s;
}

static float dotq_q4_1(const struct block_q4_1 *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		const int8_t *x = a->q + i * 32;
		int32_t t = 0;

		for (unsigned j = 0; j < 16; j++) {
			t += (int32_t)(b[i].qs[j] & 0x0f) * x[j];
			t += (int32_t)(b[i].qs[j] >> 4) * x[j + 16];
		}
		s += half_to_float(b[i].d) * a->d[i] * (float)t +
		     half_to_float(b[i].m) * a->bs[i];
	}
	return s;
}

static float dotq_q8_0(const struct block_q8_0 *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++)
		s += half_to_float(b[i].d) * a->d[i] *
		     (float)idot_n(b[i].qs, a->q + i * 32, 32);
	return s;
}

static float dotq_q6_K(const struct block_q6_K *b, const struct charsiu_act *a,
		       uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float d = half_to_float(b[i].d);

		for (unsigned n = 0; n < 2; n++) {
			const uint8_t *ql = b[i].ql + n * 64;
			const uint8_t *qh = b[i].qh + n * 32;
			const int8_t *sc = b[i].scales + n * 8;
			const int8_t *xq = a->q + i * 256 + n * 128;
			const float *xd = a->d + i * 8 + n * 4;

			for (unsigned g = 0; g < 2; g++) {
				int32_t t[4] = { 0, 0, 0, 0 };

				for (unsigned j = 0; j < 16; j++) {
					unsigned l = g * 16 + j;
					int q1 = (int8_t)((ql[l]      & 0xf) | (((qh[l] >> 0) & 3) << 4)) - 32;
					int q2 = (int8_t)((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) - 32;
					int q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
					int q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

					t[0] += q1 * xq[0 * 32 + l];
					t[1] += q2 * xq[1 * 32 + l];
					t[2] += q3 * xq[2 * 32 + l];
					t[3] += q4 * xq[3 * 32 + l];
				}
				for (unsigned k = 0; k < 4; k++)
					s += d * (float)sc[g + 2 * k] * xd[k] * (float)t[k];
			}
		}
	}
	return s;
}


/* the portable batched pair; read against the NEON ones above */
static void dotq_q4_0_m(const struct block_q4_0 *b, const struct charsiu_act *a,
			unsigned m, uint64_t nb, float *out, uint64_t ostride)
{
	unsigned j;

	for (j = 0; j < m; j++)
		out[j * ostride] = dotq_q4_0(b, &a[j], nb);
}

static void dotq_q8_0_m(const struct block_q8_0 *b, const struct charsiu_act *a,
			unsigned m, uint64_t nb, float *out, uint64_t ostride)
{
	unsigned j;

	for (j = 0; j < m; j++)
		out[j * ostride] = dotq_q8_0(b, &a[j], nb);
}

#endif

/*
 * M activations against one weight, reading the weight ONCE.
 *
 * ⚠ IT FALLS BACK PER TOKEN FOR ANYTHING IT DOES NOT HAVE A BATCHED KERNEL
 * FOR, so the answer is the same everywhere and only the speed differs. A
 * shape that quietly took a different arithmetic path would be the one bug
 * this is least able to notice.
 */
void gguf_matmul(const struct gguf_tensor *w, const struct charsiu_act *a,
		 unsigned m, float *y, uint64_t ystride,
		 uint64_t row0, uint64_t nrows)
{
	uint64_t nc = w->ne[0];
	const uint8_t *base = w->data;

	/*
	 * ⚠ THE BATCHED KERNELS BELOW ACCUMULATE INTO float s[CHARSIU_BATCH_MAX]
	 * ON THE STACK, and nothing here stopped a caller asking for more. No
	 * caller in the runtime does -- only bench_batch, which caps itself --
	 * so this has never fired, which is exactly the kind of check that is
	 * missing when it is finally needed. Split instead of smashing.
	 */
	if (m > CHARSIU_BATCH_MAX) {
		unsigned done = 0;

		while (done < m) {
			unsigned c = m - done < CHARSIU_BATCH_MAX
				   ? m - done : CHARSIU_BATCH_MAX;

			gguf_matmul(w, a + done, c, y + done * ystride, ystride,
				    row0, nrows);
			done += c;
		}
		return;
	}

	if (m == 1) {
		gguf_matvec(w, a, y, row0, nrows);
		return;
	}
	if (a->quantised) {
		switch (w->type) {
		case GGML_Q4_0:
			for (uint64_t r = 0; r < nrows; r++)
				dotq_q4_0_m((const struct block_q4_0 *)
					    (base + (row0 + r) * nc / 32 * 18),
					    a, m, nc / 32, y + row0 + r, ystride);
			return;
		case GGML_Q8_0:
			for (uint64_t r = 0; r < nrows; r++)
				dotq_q8_0_m((const struct block_q8_0 *)
					    (base + (row0 + r) * nc / 32 * 34),
					    a, m, nc / 32, y + row0 + r, ystride);
			return;
		default:
			break;
		}
	}
	for (unsigned j = 0; j < m; j++)
		gguf_matvec(w, &a[j], y + j * ystride, row0, nrows);
}

void gguf_matvec(const struct gguf_tensor *w, const struct charsiu_act *a,
		 float *y, uint64_t row0, uint64_t nrows)
{
	uint64_t nc = w->ne[0];
	const uint8_t *base = w->data;
	const float *x = a->f;

	/*
	 * A quantised weight meets the quantised activation and the dot product
	 * is integer. A float weight does not: quantising the activation for it
	 * would lose accuracy and buy nothing, since the weight has to be read
	 * as a float either way.
	 */
	if (a->quantised) {
		switch (w->type) {
		case GGML_Q8_0:
			for (uint64_t r = 0; r < nrows; r++)
				y[row0 + r] = dotq_q8_0((const struct block_q8_0 *)(base + (row0 + r) * nc / 32 * 34), a, nc / 32);
			return;
		case GGML_Q4_0:
			for (uint64_t r = 0; r < nrows; r++)
				y[row0 + r] = dotq_q4_0((const struct block_q4_0 *)(base + (row0 + r) * nc / 32 * 18), a, nc / 32);
			return;
		case GGML_Q4_1:
			for (uint64_t r = 0; r < nrows; r++)
				y[row0 + r] = dotq_q4_1((const struct block_q4_1 *)(base + (row0 + r) * nc / 32 * 20), a, nc / 32);
			return;
		case GGML_Q6_K:
			for (uint64_t r = 0; r < nrows; r++)
				y[row0 + r] = dotq_q6_K((const struct block_q6_K *)(base + (row0 + r) * nc / 256 * 210), a, nc / 256);
			return;
		default:
			break;
		}
	}

	switch (w->type) {
	case GGML_F32:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_f32((const float *)(base + (row0 + r) * nc * 4), x, nc);
		break;
	case GGML_F16:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_f16((const uint16_t *)(base + (row0 + r) * nc * 2), x, nc);
		break;
	case GGML_BF16:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_bf16((const uint16_t *)
				(base + (row0 + r) * nc * 2), x, nc);
		break;
	case GGML_Q8_0:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_q8_0((const struct block_q8_0 *)(base + (row0 + r) * nc / 32 * 34), x, nc / 32);
		break;
	case GGML_Q4_0:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_q4_0((const struct block_q4_0 *)(base + (row0 + r) * nc / 32 * 18), x, nc / 32);
		break;
	case GGML_Q4_1:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_q4_1((const struct block_q4_1 *)(base + (row0 + r) * nc / 32 * 20), x, nc / 32);
		break;
	case GGML_Q6_K:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_q6_K((const struct block_q6_K *)(base + (row0 + r) * nc / 256 * 210), x, nc / 256);
		break;
	default:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = 0.0f;
		break;
	}
}

void gguf_row_f32(const struct gguf_tensor *w, uint64_t row, float *dst)
{
	uint64_t nc = w->ne[0];
	const uint8_t *base = w->data;

	switch (w->type) {
	case GGML_F32:
		memcpy(dst, base + row * nc * 4, nc * 4);
		break;
	case GGML_F16: {
		const uint16_t *r = (const uint16_t *)(base + row * nc * 2);

		for (uint64_t i = 0; i < nc; i++)
			dst[i] = half_to_float(r[i]);
		break;
	}
	case GGML_BF16: {
		const uint16_t *r = (const uint16_t *)(base + row * nc * 2);

		for (uint64_t i = 0; i < nc; i++)
			dst[i] = bf16_to_float(r[i]);
		break;
	}
	case GGML_Q8_0: {
		const struct block_q8_0 *b = (const void *)(base + row * nc / 32 * 34);

		for (uint64_t i = 0; i < nc / 32; i++) {
			float d = half_to_float(b[i].d);

			for (unsigned j = 0; j < 32; j++)
				dst[i * 32 + j] = d * (float)b[i].qs[j];
		}
		break;
	}
	case GGML_Q4_0: {
		const struct block_q4_0 *b = (const void *)(base + row * nc / 32 * 18);

		for (uint64_t i = 0; i < nc / 32; i++) {
			float d = half_to_float(b[i].d);

			for (unsigned j = 0; j < 16; j++) {
				dst[i * 32 + j]      = d * (float)((b[i].qs[j] & 0x0f) - 8);
				dst[i * 32 + j + 16] = d * (float)((b[i].qs[j] >> 4) - 8);
			}
		}
		break;
	}
	case GGML_Q4_1: {
		const struct block_q4_1 *b = (const void *)(base + row * nc / 32 * 20);

		for (uint64_t i = 0; i < nc / 32; i++) {
			float d = half_to_float(b[i].d);
			float m = half_to_float(b[i].m);

			for (unsigned j = 0; j < 16; j++) {
				dst[i * 32 + j]      = d * (float)(b[i].qs[j] & 0x0f) + m;
				dst[i * 32 + j + 16] = d * (float)(b[i].qs[j] >> 4) + m;
			}
		}
		break;
	}
	case GGML_Q6_K: {
		const struct block_q6_K *b = (const void *)(base + row * nc / 256 * 210);

		for (uint64_t i = 0; i < nc / 256; i++)
			deq_q6_K(&b[i], dst + i * 256);
		break;
	}
	default:
		memset(dst, 0, nc * 4);
		break;
	}
}
