// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
/*
 * The fp16 attention arm's two thresholds, on a desk with no NPU.
 *
 * attn_npu_min_for() answers one question -- how long a prompt this model
 * needs before the arm pays -- out of two measured numbers and nothing in
 * between: attn_npu_min_tokens() (320) when the model shares KV heads across
 * query heads, attn_npu_min_tokens_mha() (448) when it shares none. That
 * decision had no test, and it was wrong twice in one day in two different
 * directions: once as a flat 448 for everybody, once as an outright refusal
 * for the no-GQA models. Both were plausible, both were measured, and neither
 * survived the afternoon. What a test can hold still is the SHAPE of the
 * decision -- which side of the boundary a head count lands on, which number
 * each side reads, and that a length equal to the threshold is in and one
 * below it is out -- none of which needs hardware.
 *
 * ⚠ THIS TEST DOES NOT CLAIM 320 AND 448 ARE THE RIGHT NUMBERS. They are
 * board measurements and they can move; the test pins them so that moving
 * them is a deliberate edit of two lines here and not a silent drift, and it
 * pins the two override knobs so a sweep can still name either half.
 *
 * --- why it includes the translation unit -------------------------------
 *
 * attn_npu_min_tokens(), attn_npu_min_tokens_mha() and attn_npu_min_for() are
 * static in src/llama.c, and the composed decision -- the one the runtime
 * actually makes -- is not a function at all: it is two clauses inside
 * attn_npu_get(). Three routes were open, and this takes the one that changes
 * no runtime code: the test IS a translation unit that includes src/llama.c,
 * and links against the rest of the library in llama.c's place. Adding an
 * exported shim, or lifting the two clauses into a named predicate, would
 * both have been edits to a path whose whole argument this round is that it
 * is measured; a test is not a reason to move it.
 *
 * --- what "engaged" is read from ----------------------------------------
 *
 * The refusal is defined by what it does NOT allocate. attn_npu_get's own
 * note says so: r401 put the length test one level down, inside
 * attn_npu_layer, and measured 107 ms MORE than the CPU arm at 202 tokens
 * while refusing every layer, because the mirror was still built and still
 * fed -- "a refusal that leaves the cost behind is not a refusal". So the
 * observable is s->anpu: the gate returns NULL with it still NULL when it
 * refuses, and assigns it in the same breath as the calloc when it passes.
 * That is the property the comment names, and it is the one checked here.
 *
 * ⚠ n_ctx IS 0 IN EVERY CASE ON PURPOSE. Past the gate, attn_npu_get's next
 * guard is `a->nk < 32`, and with no context that is the end of the walk --
 * no buffer objects, no device open, no stderr, on a host that has no NPU to
 * open. s->anpu has already been assigned by then, so the oracle is intact
 * and nothing downstream of the decision runs. A test of the gate must not
 * depend on what the hardware would have said after it.
 *
 * ⚠ ONE CASE PER PROCESS. Every knob below is read once into a function
 * static and cached for the life of the process, so a second case in the same
 * process would read the first case's environment. Each case is therefore a
 * fork, and the parent never calls any of these functions itself -- a fork
 * after the parent had primed a static would inherit the primed value.
 */
#include "../src/llama.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* the two numbers this tree currently ships, quoted once */
#define GQA_MIN 320u
#define MHA_MIN 448u

#define ENGAGE 1
#define REFUSE 0
#define NOMIN  0u        /* "do not check attn_npu_min_for for this case" */

struct arm {
	const char *what;
	/* NULL leaves the knob unset; "" sets it to the empty string, which
	 * is NOT the same thing for an atol knob and is checked below */
	const char *e_min;       /* CHARSIU_ATTN_NPU_MIN */
	const char *e_mha;       /* CHARSIU_ATTN_NPU_MHA_MIN */
	const char *e_npu;       /* CHARSIU_ATTN_NPU */
	unsigned n_head;
	unsigned n_head_kv;      /* 0 = the key was absent from the gguf */
	int prompt_total;
	unsigned want_min;       /* expected attn_npu_min_for(), NOMIN to skip */
	int want_engage;
};

/*
 * ⚠ THE GGUF LOADER DEFAULTS head_count_kv TO head_count -- llama_load reads
 * it as GETU("attention.head_count_kv", m->n_head_kv, m->n_head) -- so a file
 * that never states it is a no-GQA file and has to take the MHA threshold.
 * n_head_kv == 0 reaching attn_npu_min_for at all means the model was not
 * loaded through that path, and the `kvh ? kvh : n_head` fallback in
 * attn_npu_min_for has to agree with the loader. The two rows marked "unset"
 * are that agreement.
 */
static const struct arm arms[] = {
	/* ---- which threshold, from the head counts alone ---- */
	{ "gqa 4:1 takes the shared threshold",
	  NULL, NULL, NULL, 32, 8, 852, GQA_MIN, ENGAGE },
	{ "gqa 2:1 takes the shared threshold",
	  NULL, NULL, NULL, 16, 8, 852, GQA_MIN, ENGAGE },
	{ "gqa 4:1 with four heads takes the shared threshold",
	  NULL, NULL, NULL, 4, 1, 852, GQA_MIN, ENGAGE },
	{ "one kv head under many takes the shared threshold",
	  NULL, NULL, NULL, 32, 1, 852, GQA_MIN, ENGAGE },
	/* the boundary itself, from both sides and exactly on it */
	{ "n_head_kv one below n_head is still GQA",
	  NULL, NULL, NULL, 32, 31, 852, GQA_MIN, ENGAGE },
	{ "n_head_kv == n_head exactly takes the MHA threshold",
	  NULL, NULL, NULL, 32, 32, 852, MHA_MIN, ENGAGE },
	{ "n_head_kv above n_head takes the MHA threshold",
	  NULL, NULL, NULL, 32, 33, 852, MHA_MIN, ENGAGE },
	{ "one head, one kv head, takes the MHA threshold",
	  NULL, NULL, NULL, 1, 1, 852, MHA_MIN, ENGAGE },
	{ "n_head_kv unset takes the MHA threshold",
	  NULL, NULL, NULL, 32, 0, 852, MHA_MIN, ENGAGE },

	/* ---- the length, against the shared threshold ---- */
	{ "gqa, one token below 320, refuses",
	  NULL, NULL, NULL, 32, 8, 319, GQA_MIN, REFUSE },
	{ "gqa, exactly 320, engages",
	  NULL, NULL, NULL, 32, 8, 320, GQA_MIN, ENGAGE },
	{ "gqa, one token above 320, engages",
	  NULL, NULL, NULL, 32, 8, 321, GQA_MIN, ENGAGE },
	/*
	 * ⚠ 352 AND 452 ARE THE TWO MEASURED LENGTHS, and they are the reason
	 * the two numbers are not one. r412 read both models at 352: the two
	 * that share KV heads won there, the two that do not lost by 15.9%
	 * and 11.2%. A single threshold cannot express that, and these two
	 * rows are the arithmetic of it.
	 */
	{ "gqa at 352, the measured win, engages",
	  NULL, NULL, NULL, 32, 8, 352, GQA_MIN, ENGAGE },
	{ "no-GQA at 352, the measured loss, refuses",
	  NULL, NULL, NULL, 32, 32, 352, MHA_MIN, REFUSE },

	/* ---- the length, against the MHA threshold ---- */
	{ "no-GQA, one token below 448, refuses",
	  NULL, NULL, NULL, 32, 32, 447, MHA_MIN, REFUSE },
	{ "no-GQA, exactly 448, engages",
	  NULL, NULL, NULL, 32, 32, 448, MHA_MIN, ENGAGE },
	{ "no-GQA, one token above 448, engages",
	  NULL, NULL, NULL, 32, 32, 449, MHA_MIN, ENGAGE },
	{ "no-GQA at 452, the measured win, engages",
	  NULL, NULL, NULL, 32, 32, 452, MHA_MIN, ENGAGE },
	{ "n_head_kv unset, below 448, refuses like MHA",
	  NULL, NULL, NULL, 32, 0, 447, MHA_MIN, REFUSE },

	/* ---- no hint at all ---- */
	/*
	 * ⚠ 0 IS NOT A SHORT PROMPT, IT IS NO ANSWER. llama_prefill_hint is
	 * what fills prompt_total, and a tool that never calls it leaves 0
	 * behind. auto has to read that as "I was not told" and stay on the
	 * CPU arm, not as "this prompt is zero tokens long", which would be
	 * below every threshold anyway -- but a negative one would not be, if
	 * the comparison were ever written unsigned without the guard.
	 */
	{ "no hint (0) refuses even for GQA",
	  NULL, NULL, NULL, 32, 8, 0, GQA_MIN, REFUSE },
	{ "no hint (0) refuses for no-GQA",
	  NULL, NULL, NULL, 32, 32, 0, MHA_MIN, REFUSE },
	{ "a negative hint refuses",
	  NULL, NULL, NULL, 32, 8, -1, GQA_MIN, REFUSE },
	{ "a very negative hint refuses",
	  NULL, NULL, NULL, 32, 8, -1000000, GQA_MIN, REFUSE },

	/* ---- the overrides, and that each moves only its own half ---- */
	{ "CHARSIU_ATTN_NPU_MIN moves the shared threshold",
	  "64", NULL, NULL, 32, 8, 64, 64u, ENGAGE },
	{ "CHARSIU_ATTN_NPU_MIN, one below, refuses",
	  "64", NULL, NULL, 32, 8, 63, 64u, REFUSE },
	{ "CHARSIU_ATTN_NPU_MIN does not move the MHA threshold",
	  "64", NULL, NULL, 32, 32, 447, MHA_MIN, REFUSE },
	{ "CHARSIU_ATTN_NPU_MHA_MIN moves the MHA threshold",
	  NULL, "64", NULL, 32, 32, 64, 64u, ENGAGE },
	{ "CHARSIU_ATTN_NPU_MHA_MIN, one below, refuses",
	  NULL, "64", NULL, 32, 32, 63, 64u, REFUSE },
	{ "CHARSIU_ATTN_NPU_MHA_MIN does not move the shared threshold",
	  NULL, "64", NULL, 32, 8, 319, GQA_MIN, REFUSE },
	{ "both knobs at once, each on its own model: GQA",
	  "600", "64", NULL, 32, 8, 599, 600u, REFUSE },
	{ "both knobs at once, each on its own model: no-GQA",
	  "600", "64", NULL, 32, 32, 64, 64u, ENGAGE },
	/*
	 * ⚠ 0 HERE MEANS NO MINIMUM, NOT OFF. Whoever sets
	 * CHARSIU_ATTN_NPU_MIN=0 expecting the arm to stop gets it on at every
	 * length that has a hint. CHARSIU_ATTN_NPU=0 is the off switch; these
	 * two rows are the difference, and they are the reason the off switch
	 * is a separate knob.
	 */
	{ "CHARSIU_ATTN_NPU_MIN=0 is no floor, not off",
	  "0", NULL, NULL, 32, 8, 1, 0u, ENGAGE },
	{ "CHARSIU_ATTN_NPU_MIN=0 still refuses with no hint",
	  "0", NULL, NULL, 32, 8, 0, 0u, REFUSE },
	{ "CHARSIU_ATTN_NPU_MHA_MIN=0 is no floor, not off",
	  NULL, "0", NULL, 32, 32, 1, 0u, ENGAGE },
	/* a negative override clamps to no floor rather than wrapping to 4
	 * billion, which is what the unsigned cast would have made of it */
	{ "a negative CHARSIU_ATTN_NPU_MIN clamps to 0",
	  "-1", NULL, NULL, 32, 8, 1, 0u, ENGAGE },
	{ "a negative CHARSIU_ATTN_NPU_MHA_MIN clamps to 0",
	  NULL, "-5", NULL, 32, 32, 1, 0u, ENGAGE },
	/*
	 * ⚠ SET BUT EMPTY IS THE DEFAULT FOR THESE TWO, and it is NOT for the
	 * flag knobs next to them: charsiu_env_flag reads an empty value as 0,
	 * these read it as "nothing said". `FOO= cmd` is therefore off for one
	 * and default for the other, and this row is which.
	 */
	{ "CHARSIU_ATTN_NPU_MIN set empty is the default",
	  "", NULL, NULL, 32, 8, 319, GQA_MIN, REFUSE },
	{ "CHARSIU_ATTN_NPU_MHA_MIN set empty is the default",
	  NULL, "", NULL, 32, 32, 447, MHA_MIN, REFUSE },

	/* ---- the enable knob sits above both thresholds ---- */
	{ "CHARSIU_ATTN_NPU=0 refuses a prompt past both thresholds",
	  NULL, NULL, "0", 32, 8, 852, GQA_MIN, REFUSE },
	{ "CHARSIU_ATTN_NPU=0 refuses a no-GQA prompt past 448",
	  NULL, NULL, "0", 32, 32, 852, MHA_MIN, REFUSE },
	/* ⚠ =1 does not consult the hint at all, which is what makes it a
	 * usable control arm: it is on at every length, including no hint */
	{ "CHARSIU_ATTN_NPU=1 engages below the shared threshold",
	  NULL, NULL, "1", 32, 8, 1, GQA_MIN, ENGAGE },
	{ "CHARSIU_ATTN_NPU=1 engages with no hint at all",
	  NULL, NULL, "1", 32, 32, 0, MHA_MIN, ENGAGE },
	{ "CHARSIU_ATTN_NPU=auto is the default, and refuses below 320",
	  NULL, NULL, "auto", 32, 8, 319, GQA_MIN, REFUSE },
	{ "CHARSIU_ATTN_NPU=auto engages at 320",
	  NULL, NULL, "auto", 32, 8, 320, GQA_MIN, ENGAGE },
	/*
	 * ⛔ AND =2 IS NOT auto, WHICH THE COMMENT OVER attn_npu_want_for
	 * SAYS IT IS. That comment's last line reads "0 off, 1 on for every
	 * layer, 2 decide per prompt on its length", and the code under it
	 * collapses every value that is neither empty nor the word `auto` to
	 * `atoi(e) != 0` -- so 2, 3 and 448 are all "on for every layer" and
	 * only the literal string `auto` and an unset variable reach the
	 * length test. This row asserts what the code does, because that is
	 * what a test is for; the comment is reported as wrong rather than
	 * quietly satisfied here.
	 */
	{ "CHARSIU_ATTN_NPU=2 is forced on, not auto",
	  NULL, NULL, "2", 32, 32, 447, MHA_MIN, ENGAGE },
	/* ⚠ and set-but-empty is auto here too, not off: `CHARSIU_ATTN_NPU=
	 * charsiu_run` is the DEFAULT, where the same spelling on any
	 * charsiu_env_flag knob in this file means 0 */
	{ "CHARSIU_ATTN_NPU set empty is auto, and refuses below 448",
	  NULL, NULL, "", 32, 32, 447, MHA_MIN, REFUSE },
	{ "CHARSIU_ATTN_NPU set empty is auto, and engages at 448",
	  NULL, NULL, "", 32, 32, 448, MHA_MIN, ENGAGE },
};

static void set_or_clear(const char *name, const char *v)
{
	if (v)
		setenv(name, v, 1);
	else
		unsetenv(name);
}

/* runs in the child, with a virgin set of function statics */
static int run_arm(const struct arm *a)
{
	struct llama_model m;
	struct llama_state *s;
	unsigned got_min;
	int engaged, bad = 0;

	set_or_clear("CHARSIU_ATTN_NPU_MIN", a->e_min);
	set_or_clear("CHARSIU_ATTN_NPU_MHA_MIN", a->e_mha);
	set_or_clear("CHARSIU_ATTN_NPU", a->e_npu);

	memset(&m, 0, sizeof(m));
	m.n_head = a->n_head;
	m.n_head_kv = a->n_head_kv;
	m.head_dim = 64;
	m.n_embd = a->n_head * 64;
	m.n_layer = 1;

	if (a->want_min != NOMIN || a->e_min || a->e_mha) {
		got_min = attn_npu_min_for(&m);
		if (got_min != a->want_min) {
			printf("  %s: attn_npu_min_for is %u, expected %u "
			       "(n_head %u, n_head_kv %u)\n",
			       a->what, got_min, a->want_min,
			       a->n_head, a->n_head_kv);
			bad = 1;
		}
	}

	/*
	 * ⚠ THE COMPOSED DECISION, THROUGH THE DOOR THE RUNTIME USES.
	 * charsiu_run calls llama_prefill_hint and then the forward pass
	 * reaches attn_npu_get; so does this. n_ctx stays 0, which stops the
	 * walk at the guard immediately after the gate.
	 */
	s = calloc(1, sizeof(*s));
	if (!s) {
		printf("  %s: out of memory\n", a->what);
		return 1;
	}
	s->m = &m;
	s->n_ctx = 0;
	llama_prefill_hint(s, a->prompt_total);

	attn_npu_get(s);
	engaged = s->anpu != NULL;
	if (engaged != a->want_engage) {
		printf("  %s: the gate %s at %d tokens "
		       "(n_head %u, n_head_kv %u, threshold %u), expected %s\n",
		       a->what, engaged ? "engaged" : "refused",
		       a->prompt_total, a->n_head, a->n_head_kv,
		       attn_npu_min_for(&m),
		       a->want_engage ? "engage" : "refuse");
		bad = 1;
	}
	free(s->anpu);
	free(s);
	return bad;
}

int main(void)
{
	const unsigned n = sizeof(arms) / sizeof(arms[0]);
	unsigned i;
	int fail = 0;

	for (i = 0; i < n; i++) {
		pid_t p;
		int st = 0;

		/* ⚠ FLUSH BEFORE THE FORK, or the child inherits whatever is
		 * still in the parent's buffer and prints it a second time */
		fflush(stdout);
		p = fork();
		if (p < 0) {
			printf("  fork failed\n");
			return 1;
		}
		if (!p) {
			/* ⚠ AND FLUSH AFTER THE ARM, NOT BEFORE IT. _exit does
			 * not flush, so a diagnosis written by run_arm is lost
			 * unless it is pushed out here -- which is a failing
			 * test that says nothing about why. */
			int bad = run_arm(&arms[i]);

			fflush(stdout);
			_exit(bad ? 1 : 0);
		}
		if (waitpid(p, &st, 0) < 0 || !WIFEXITED(st)
		    || WEXITSTATUS(st)) {
			if (WIFSIGNALED(st))
				printf("  %s: died on signal %d\n",
				       arms[i].what, WTERMSIG(st));
			fail++;
		}
	}

	/*
	 * ⚠ AND THE TWO NUMBERS THEMSELVES, once, in a process of their own.
	 * Every row above is a shape; this is the only place the shipped
	 * values appear as values, so moving one is a one line edit here and
	 * not a test that quietly keeps passing.
	 */
	{
		pid_t p;
		int st = 0;

		fflush(stdout);
		p = fork();
		if (!p) {
			unsigned g, h;

			unsetenv("CHARSIU_ATTN_NPU_MIN");
			unsetenv("CHARSIU_ATTN_NPU_MHA_MIN");
			g = attn_npu_min_tokens();
			h = attn_npu_min_tokens_mha();
			if (g != GQA_MIN || h != MHA_MIN)
				printf("  the shipped thresholds are %u/%u, "
				       "this test was written against %u/%u\n",
				       g, h, GQA_MIN, MHA_MIN);
			fflush(stdout);
			_exit(g == GQA_MIN && h == MHA_MIN ? 0 : 1);
		}
		if (p < 0 || waitpid(p, &st, 0) < 0 || !WIFEXITED(st)
		    || WEXITSTATUS(st))
			fail++;
	}

	printf("  attention thresholds: %d of %u cases wrong\n", fail, n + 1);
	return fail ? 1 : 0;
}
