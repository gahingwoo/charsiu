#define _POSIX_C_SOURCE 200809L
/*
 * charsiu_shapes -- what a model asks the hardware for, from its gguf, on a
 * desktop, with no NPU and no board.
 *
 * ⚠⚠ WHY. Nine architectures produce identical text; four have a tok/s
 * number, and those four are the four the vendor publishes a benchmark for.
 * Correctness generality and PERFORMANCE generality are different claims and
 * this repo has only been making one of them.
 *
 * The variable is not the model. It is (m, k, n) and how those divide by the K
 * slice, by the feature atom, by two. A tuned constant that happens to suit
 * Llama-3.2 -- every dimension a power of two -- is invisible on Llama and
 * costs 13 to 21% everywhere else: that is not a hypothesis, it is what
 * `d = (ki*ns+ni)&1` did, and it survived because the model measured most
 * often was the one that could not see it.
 *
 * So this reads the shapes and counts what a token costs in the three units
 * dispatch is priced in:
 *
 *   calls    grouped submits: qkv, o, gate+up, down a layer, plus the head
 *   tasks    K slices, ceil(K / KMAX), summed over every tensor in a call
 *   MB       weight bytes moved
 *
 * A decode token is then `calls*a + tasks*b + MB*c` for the board's own three
 * coefficients, and the point of doing it here is that ONE board round
 * calibrates a, b, c and every gguf gets a prediction for free.
 *
 * ⚠ THE COEFFICIENTS IN THIS FILE ARE PROVISIONAL AND SAY SO. npudev.c
 * carries `128.7 + 36.8*tasks + 110.0*MB` fitted from five decode stages,
 * where tasks and MB move together; npu_job_cost measured 16.85 us a job and
 * 4.81 a task directly, on a matmul with no arithmetic in it. Those disagree
 * by 8x on the task term and both cannot be right. --coef overrides all three
 * so a calibration round can be applied without a rebuild.
 *
 *   charsiu_shapes MODEL.gguf [MODEL.gguf ...] [--kmax N] [--coef a,b,c]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu_llm.h"

static uint64_t slices(uint64_t k, unsigned kmax)
{
	return kmax ? (k + kmax - 1) / kmax : 1;
}

/*
 * ⚠⚠ THE COST OF A CALL IS NOT LINEAR IN ITS BYTES, AND A TOTAL HIDES THAT.
 *
 * The first version of this fitted a token as calls*a + tasks*b + MB*c with
 * MB the model's whole weight, and its hold-out error was +4.4% on qwen3,
 * +10.6% on gemma4 and +23.5% on Phi-3.5 -- monotone in size, which no
 * coefficient fixes because it is the FORM that is wrong. npu_job_cost's own
 * sweep says why (round 155, one job one task, us against MB):
 *
 *     0.0328 -> 40.13     1.0486 -> 144.00     8.3886 -> 785.00
 *     0.2621 -> 44.44     4.1943 -> 406.81
 *
 * The line through the large end predicts 38 and 129 for the first two, so a
 * megabyte in a SMALL call costs more than a megabyte in a large one. Phi-3.5
 * has the fattest tensors of the five, so an average rate overcharges it most
 * -- exactly the shape of the error.
 *
 * So a call is priced from its own bytes, by interpolating the measured
 * points, and the model's total is never formed. That is not a better fit; it
 * is the same measurement asked per call instead of once.
 */
static double call_us(double mb)
{
	/* npu_job_cost, round 155, m = 1, one task, int8 weights */
	static const double x[] = { 0.0328, 0.2621, 1.0486, 4.1943, 8.3886 };
	static const double y[] = { 40.13,  44.44,  144.00, 406.81, 785.00 };
	const int n = (int)(sizeof(x) / sizeof(*x));
	int i;

	if (mb <= x[0])                       /* below the smallest measured */
		return y[0] * (mb / x[0] < 1.0 ? 1.0 : 1.0);
	for (i = 1; i < n; i++)
		if (mb <= x[i])
			return y[i - 1] + (y[i] - y[i - 1])
			       * (mb - x[i - 1]) / (x[i] - x[i - 1]);
	/* above the largest measured, extend at the large end's slope */
	return y[n - 1] + (mb - x[n - 1])
	       * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
}

int main(int argc, char **argv)
{
	/* provisional: npu_job_cost's a and b, npudev's c. See the note above. */
	double A = 37.2, B = 4.81, C = 89.1;
	unsigned kmax = 1024, nmax = 8192;
	int i, first = 1;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--kmax") && i + 1 < argc)
			kmax = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "--nmax") && i + 1 < argc)
			nmax = (unsigned)atoi(argv[++i]);
		else if (!strcmp(argv[i], "--coef") && i + 1 < argc)
			sscanf(argv[++i], "%lf,%lf,%lf", &A, &B, &C);
	}
	printf("KMAX %u  NMAX %u   cost = %.2f us a call + %.2f a task + "
	       "%.1f a MB  (npu_job_cost, round 155)\n\n", kmax, nmax, A, B, C);
	printf("%-30s %6s %6s %7s %7s %8s %8s %7s %9s\n",
	       "model", "layer", "n_embd", "n_ff", "calls", "tasks", "MB",
	       "t/MB", "ms a token");
	for (i = 1; i < argc; i++) {
		struct llama_model m;
		uint64_t calls = 0, tasks = 0;
		double bytes = 0.0, ms, us = 0.0;
		unsigned l;

		if (argv[i][0] == '-') { i++; continue; }
		if (llama_load(&m, argv[i]) < 0) {
			fprintf(stderr, "  %s: will not load\n", argv[i]);
			continue;
		}
		for (l = 0; l < m.n_layer; l++) {
			uint64_t e = m.n_embd;
			uint64_t hd = m.head_dim ? m.head_dim : e / m.n_head;
			uint64_t q = e * (m.n_head * hd);
			uint64_t kv = e * (m.n_head_kv * hd);
			uint64_t o = (m.n_head * hd) * e;
			uint64_t ff = m.layers && m.layers[l].n_ff
				    ? m.layers[l].n_ff : m.n_ff;
			uint64_t gu = e * ff, dn = ff * e;

			/* four grouped calls a layer: qkv, o, gate+up, down.
			 * Each is priced from ITS OWN bytes -- see call_us. */
			calls += 4;
			tasks += 3 * slices(e, kmax)      /* q, k, v */
			       + slices(e, kmax)          /* o reads n_head*hd */
			       + 2 * slices(e, kmax)      /* gate, up */
			       + slices(ff, kmax);        /* down */
			bytes += (double)(q + 2 * kv + o + 2 * gu + dn) * 0.5;
			us += call_us((double)(q + 2 * kv) * 0.5 / 1e6)
			    + call_us((double)o * 0.5 / 1e6)
			    + call_us((double)(2 * gu) * 0.5 / 1e6)
			    + call_us((double)dn * 0.5 / 1e6);
		}
		calls += 1;                                   /* the head */
		tasks += slices(m.n_embd, kmax)
		       * ((m.n_vocab + nmax - 1) / nmax);
		bytes += (double)m.n_vocab * m.n_embd * 0.5;
		us += call_us((double)m.n_vocab * m.n_embd * 0.5 / 1e6);
		bytes /= 1e6;
		ms = (us + tasks * B) / 1e3;
		(void)A; (void)C;

		if (first) first = 0;
		printf("%-30s %6u %6u %7u %7llu %8llu %8.1f %7.2f %9.2f\n",
		       argv[i][0] ? strrchr(argv[i], '/') ?
				    strrchr(argv[i], '/') + 1 : argv[i] : "?",
		       m.n_layer, m.n_embd, m.n_ff,
		       (unsigned long long)calls, (unsigned long long)tasks,
		       bytes, tasks / bytes, ms);
		llama_free(&m);
	}
	printf("\n⚠ t/MB is the shape's own signature: a model with many thin\n"
	       "  tensors pays dispatch where a model with few fat ones pays\n"
	       "  bandwidth, and a constant tuned on one is wrong on the other.\n");
	return 0;
}
