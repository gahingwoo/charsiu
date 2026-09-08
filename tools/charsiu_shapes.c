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
/*
 * ⚠⚠ A CALL IS TWO JOBS, AND THE PROBE MEASURED ONE.
 *
 * npu_job_cost times a single job on a single fd. charsiu issues one per core
 * and waits on both, so a call moves its bytes through two devices. Pricing it
 * as one job overestimates, and by more the fatter the tensor -- which is
 * exactly the hold-out's shape: +8.4% on qwen3, +19.8% on gemma-3-1b, +34.5%
 * on Phi-3.5, all high, monotone in size.
 *
 * The two cores are not worth two, and that is measured rather than assumed:
 * round 150 ran the same decode on one core and on two, 39.6 against 30.8 ms a
 * token, so the pair is worth **1.286x**. Dividing each call's bytes by that
 * takes the hold-out from a 34.5% systematic overestimate to +6.0 / -3.9 /
 * -11.1% -- no longer one-sided, and no longer growing with the model.
 *
 * ⚠ CHARSIU_SHAPES_CORES overrides it, because 1.286 is one board's number at
 * one governor and it will move. It is not a fitted parameter: sweeping it
 * (1.0, 1.15, 1.286, 1.4, 1.6, 2.0) has its minimum AT the measured value,
 * which is the only reason to trust it as a mechanism rather than a knob.
 *
 * ⚠ What is still missing and is not this: the predictor counts MATMULS only,
 * and round 147 measured those at 87.8% of a qwen3 token and 90.6% of a
 * tinyllama one. Attention, the norms and the elementwise joins are 9 to 12%
 * that this number does not contain.
 */
static double core_pair(void)
{
	const char *e = getenv("CHARSIU_SHAPES_CORES");
	double v = e && *e ? atof(e) : 1.286;

	return v > 0.0 ? v : 1.286;
}

static double call_us(double mb)
{
	/*
	 * npu_job_cost, round 163, **with CHARSIU_JOB_GAP_US=40**, mean of five.
	 *
	 * ⚠⚠ THE GAP IS THE WHOLE POINT. The probe's loop submits the next job
	 * the instant the previous prep returns, and consecutive dispatches
	 * then overlap. A decode cannot: between two calls it has to rmsnorm,
	 * rope, run attention or a residual, which is tens of microseconds of
	 * real work on the same cores. Measured both ways on the same round:
	 *
	 *      MB      gap 0     gap 40
	 *   0.1311     33.15      79.13     2.4x
	 *   0.5243     64.80     107.96     1.7x
	 *   1.0486    150.98     153.64     1.0x
	 *  33.5544   2825.77    2829.00     1.0x
	 *
	 * Small calls are 2.4x more expensive when nothing pipelines them and
	 * large ones do not care. Every model's small calls are exactly where
	 * this predictor was low, so it was calibrated against an overlap the
	 * product never gets.
	 *
	 * ⚠ And round 162's "two shapes disagree with themselves by 90.9% and
	 * 36.9%" did not reproduce -- both read 3.3% and 1.5% here. I gave a
	 * random pair of cells a shape explanation. The instability is the
	 * first points measured in a pass, whatever they are.
	 */
	static const double x[] = { 0.0020, 0.0328, 0.1311, 0.2621, 0.5243,
				    1.0486, 4.1943, 8.3886, 16.7772, 33.5544 };
	static const double y[] = { 66.13,  68.42,  79.13,  80.31,  107.96,
				    153.64, 371.38, 776.53, 1473.48, 2829.00 };
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
			us += call_us((double)(q + 2 * kv) * 0.5 / 1e6 / core_pair())
			    + call_us((double)o * 0.5 / 1e6 / core_pair())
			    + call_us((double)(2 * gu) * 0.5 / 1e6 / core_pair())
			    + call_us((double)dn * 0.5 / 1e6 / core_pair());
		}
		/*
		 * ⚠⚠ THE HEAD IS NOT ONE 156 MB CALL, IT IS ceil(n_vocab/NMAX)
		 * SLICES OF AT MOST NMAX.
		 *
		 * Pricing it whole asked call_us for 38 MB on Phi-3.5 and 157
		 * on gemma4, both far outside the sweep, so the answer was a
		 * straight-line extrapolation at 89 us a megabyte -- while
		 * round 147 measured the head itself at 15.56 GB/s, which is
		 * 64. Three models then came back implying a matmul share above
		 * 100% of their own token, which is the arithmetic refusing the
		 * extrapolation rather than a fit being poor.
		 *
		 * A slice is at most NMAX wide by construction, so every piece
		 * lands inside the measured range and nothing is extrapolated.
		 */
		{
			unsigned nsl = (m.n_vocab + nmax - 1) / nmax;
			unsigned left = m.n_vocab, w;

			calls += 1;                           /* the head */
			tasks += slices(m.n_embd, kmax) * nsl;
			bytes += (double)m.n_vocab * m.n_embd * 0.5;
			while (left) {
				w = left > nmax ? nmax : left;
				us += call_us((double)w * m.n_embd * 0.5
					      / 1e6 / core_pair());
				left -= w;
			}
		}
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
