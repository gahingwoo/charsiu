/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * Does batching the projections pay on THIS machine?
 *
 * ⚠ IT HAS TO WALK EVERY LAYER. The first version of this looped on one
 * tensor 200 times, which left it in cache and measured arithmetic rather
 * than memory: it reported 1.09x where the question is entirely about how
 * often the weights are read from DRAM. Walking the model streams them the
 * way a forward pass does.
 *
 * On the development host (a big cache, wide memory) this says 1.1x to 1.4x.
 * The board is 10 GB/s of LPDDR5 behind four A72s, where the same kernel is
 * far closer to the memory roof, so the number that decides whether a batched
 * forward is worth building is the one THIS prints on the board.
 *
 *   bench_batch MODEL.gguf M
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "charsiu_llm.h"
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
/* Walk EVERY layer, the way a forward pass does, so the weights are streamed
 * from memory instead of sitting in cache like a single-tensor loop. */
int main(int argc, char **argv)
{
	(void)argc;
	struct llama_model m; struct charsiu_act a[16]; unsigned M=atoi(argv[2]),j;
	if(llama_load(&m,argv[1])<0)return 1;
	uint64_t kmax=0,nmax=0;
	for(uint32_t l=0;l<m.n_layer;l++){
		const struct gguf_tensor*t[4]={m.layers[l].wq,m.layers[l].wo,m.layers[l].gate,m.layers[l].down};
		for(int i=0;i<4;i++){if(t[i]->ne[0]>kmax)kmax=t[i]->ne[0];if(t[i]->ne[1]>nmax)nmax=t[i]->ne[1];}}
	float *y=calloc(M*nmax,sizeof(float));
	float *x=malloc(kmax*sizeof(float));
	for(unsigned i=0;i<kmax;i++)x[i]=(float)(i%17)/8.0f-1.0f;
	for(j=0;j<M;j++){memset(&a[j],0,sizeof(a[j]));charsiu_act_alloc(&a[j],(int)kmax);}
	double t0,t1; int R=3;
	t0=ms();
	for(int r=0;r<R;r++)for(uint32_t l=0;l<m.n_layer;l++){
		const struct gguf_tensor*t[4]={m.layers[l].wq,m.layers[l].wo,m.layers[l].gate,m.layers[l].down};
		for(int i=0;i<4;i++){for(j=0;j<M;j++){charsiu_act_set(&a[j],x,(int)t[i]->ne[0]);charsiu_act_blocks(&a[j]);
			gguf_matvec(t[i],&a[j],y+j*nmax,0,t[i]->ne[1]);}}}
	t1=ms(); printf("  one token at a time : %7.1f ms\n",(t1-t0)/R);
	t0=ms();
	for(int r=0;r<R;r++)for(uint32_t l=0;l<m.n_layer;l++){
		const struct gguf_tensor*t[4]={m.layers[l].wq,m.layers[l].wo,m.layers[l].gate,m.layers[l].down};
		for(int i=0;i<4;i++){for(j=0;j<M;j++){charsiu_act_set(&a[j],x,(int)t[i]->ne[0]);charsiu_act_blocks(&a[j]);}
			gguf_matmul(t[i],a,M,y,nmax,0,t[i]->ne[1]);}}
	t1=ms(); printf("  batched             : %7.1f ms\n",(t1-t0)/R);
	return 0;}
