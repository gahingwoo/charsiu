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

static const struct type_traits g_traits[] = {
	[GGML_F32]  = { "f32",  1,   4 },
	[GGML_F16]  = { "f16",  1,   2 },
	[GGML_Q4_0] = { "q4_0", 32, 18 },
	[GGML_Q4_1] = { "q4_1", 32, 20 },
	[GGML_Q8_0] = { "q8_0", 32, 34 },
	[GGML_Q6_K] = { "q6_K", 256, 210 },
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

/* sixteen int8 against sixteen floats, widened rather than dotted */
static inline float32x4_t fma_i8x16(float32x4_t acc[4], int8x16_t q, const float *x)
{
	int16x8_t l = vmovl_s8(vget_low_s8(q));
	int16x8_t h = vmovl_s8(vget_high_s8(q));

	acc[0] = vfmaq_f32(acc[0], vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))),  vld1q_f32(x));
	acc[1] = vfmaq_f32(acc[1], vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))), vld1q_f32(x + 4));
	acc[2] = vfmaq_f32(acc[2], vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),  vld1q_f32(x + 8));
	acc[3] = vfmaq_f32(acc[3], vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))), vld1q_f32(x + 12));
	return acc[0];
}

static float dot_q8_0(const struct block_q8_0 *b, const float *x, uint64_t nb)
{
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float32x4_t acc[4] = { vdupq_n_f32(0), vdupq_n_f32(0),
				       vdupq_n_f32(0), vdupq_n_f32(0) };

		fma_i8x16(acc, vld1q_s8(b[i].qs),      x + i * 32);
		fma_i8x16(acc, vld1q_s8(b[i].qs + 16), x + i * 32 + 16);

		s += half_to_float(b[i].d) *
		     vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]),
					  vaddq_f32(acc[2], acc[3])));
	}
	return s;
}

static float dot_q4_0(const struct block_q4_0 *b, const float *x, uint64_t nb)
{
	const int8x16_t bias = vdupq_n_s8(8);
	const uint8x16_t mask = vdupq_n_u8(0x0f);
	float s = 0.0f;

	for (uint64_t i = 0; i < nb; i++) {
		float32x4_t acc[4] = { vdupq_n_f32(0), vdupq_n_f32(0),
				       vdupq_n_f32(0), vdupq_n_f32(0) };
		uint8x16_t p = vld1q_u8(b[i].qs);

		/* the low nibbles are elements 0..15, the high ones 16..31 */
		fma_i8x16(acc, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p, mask)), bias),
			  x + i * 32);
		fma_i8x16(acc, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p, 4)), bias),
			  x + i * 32 + 16);

		s += half_to_float(b[i].d) *
		     vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]),
					  vaddq_f32(acc[2], acc[3])));
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

#endif

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

void gguf_matvec(const struct gguf_tensor *w, const float *x, float *y,
		 uint64_t row0, uint64_t nrows)
{
	uint64_t nc = w->ne[0];
	const uint8_t *base = w->data;

	switch (w->type) {
	case GGML_F32:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_f32((const float *)(base + (row0 + r) * nc * 4), x, nc);
		break;
	case GGML_F16:
		for (uint64_t r = 0; r < nrows; r++)
			y[row0 + r] = dot_f16((const uint16_t *)(base + (row0 + r) * nc * 2), x, nc);
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
