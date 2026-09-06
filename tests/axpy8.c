/* A COPY of the two kernels, which is exactly what this can and cannot check:
 * it CANNOT prove the shipped code is right, but it CAN catch the likely bug in
 * a new one -- an index. The v pointers in attn_axpy8 are built from kstride
 * inside the callee where attn_axpy4 had them built by the caller. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void axpy4(float *out, const float *a, const float *v0, const float *v1,
		  const float *v2, const float *v3, unsigned hd)
{
	for (unsigned i = 0; i < hd; i++) {
		float o = out[i];
		o += a[0]*v0[i]; o += a[1]*v1[i]; o += a[2]*v2[i]; o += a[3]*v3[i];
		out[i] = o;
	}
}
static void axpy8(float *out, const float *a, const float *v, size_t ks, unsigned hd)
{
	for (unsigned i = 0; i < hd; i++) {
		float o = out[i];
		for (unsigned q = 0; q < 8; q++)
			o += a[q] * v[(size_t)q * ks + i];
		out[i] = o;
	}
}
int main(void)
{
	unsigned bad = 0, cases = 0;
	for (unsigned hd = 1; hd <= 130; hd++)
	  for (unsigned ks = hd; ks <= hd + 3; ks++) {
		float *v = malloc(sizeof(float) * ks * 8);
		float a[8], o4[256], o8[256];
		for (unsigned i = 0; i < ks * 8; i++) v[i] = (float)((i * 2654435761u) % 1000) / 997.0f - 0.5f;
		for (unsigned q = 0; q < 8; q++) a[q] = (float)((q * 40503u + hd) % 991) / 499.0f - 1.0f;
		for (unsigned i = 0; i < hd; i++) o4[i] = o8[i] = (float)(i % 17) * 0.125f;
		/* two four-wide groups must equal one eight-wide group */
		axpy4(o4, a,     v,           v + ks,     v + 2*ks, v + 3*ks, hd);
		axpy4(o4, a + 4, v + 4*ks,    v + 5*ks,   v + 6*ks, v + 7*ks, hd);
		axpy8(o8, a, v, ks, hd);
		for (unsigned i = 0; i < hd; i++)
			if (memcmp(&o4[i], &o8[i], sizeof(float))) { bad++; break; }
		cases++;
		free(v);
	  }
	printf("axpy8 vs two axpy4: %u of %u cases differ in a single bit\n", bad, cases);
	return bad != 0;
}
