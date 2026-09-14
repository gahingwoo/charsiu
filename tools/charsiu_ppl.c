/*
 * charsiu_ppl -- perplexity, so a quantiser change stops being a matter of
 * reading one paragraph and deciding it looks fine.
 *
 * THAT IS TRUE ONLY OF THE PLAIN DEFAULT, AND THIS PARAGRAPH USED TO SAY
 * OTHERWISE. It read "IT HAS TO RUN ON THE BOARD. The host has no NPU, so
 * npu_get returns NULL and every matvec falls back to the gguf weights --
 * which measures llama.cpp's q4_0 and not charsiu's int4 at all." The last
 * clause is wrong, and the disproof is one measurement: if the host were
 * scoring llama.cpp's q4_0, setting CHARSIU_NPU_W4_GROUP could not move the
 * number, and it moves it 18.6%.
 *
 * With CHARSIU_NPU_QUANT=1 the host builds charsiu's own quantised copy
 * through charsiu_pool_get with dev == NULL and matvec_again uses it, so the
 * weights in the loop ARE charsiu's int4. What the host does not reproduce is
 * the ACTIVATION: the board's int4 decode packs the real activation as fp16
 * (npudev.c, charsiu_pack_input_f16) and accumulates slices in float, while
 * npu_matvec here takes the int8 activation unless CHARSIU_NPU_A16=1. The
 * board is w4a16; this defaults to w4a8.
 *
 * SO THE HOST REPRODUCES THE BOARD, AND NOT MERELY TO A FEW PERCENT.
 * Measured on r413 with the model and the corpus checked by md5 to be the same
 * FILES on both sides (c82c0340d974 and 4237c8fc3163, long.txt, -n 300):
 *
 *     host  CPU, group 1024, w4a8   33.8071    <- the default arm
 *     BOARD real NPU, int4          32.8025
 *     host  CPU, group 1024, w4a16  32.8008    <- 0.005% from the board
 *
 * Set CHARSIU_NPU_W4_GROUP=1024 and CHARSIU_NPU_A16=1 and the desk answers the
 * board to five significant figures. Quality work does not need the card at
 * all; what it needs is both of those set, because the board is w4a16 and this
 * defaults to w4a8.
 *
 * THE FIRST VERSION OF THIS PARAGRAPH SAID "BRACKETS ... TO ABOUT 1 TO 4%",
 * comparing against a board figure of 33.4149 taken from a different round
 * under conditions nobody had matched. The bracket is real but it is an
 * artefact of the DEFAULT activation width, not the limit of what the desk can
 * do. A number belongs to its input file, and that is also true of the number
 * you are comparing against.
 *
 * AND THE GROUP IS THE WHOLE OF IT, NOT KMAX. The quantiser's group is
 * CHARSIU_NPU_W4_GROUP (npuquant.c); CHARSIU_NPU_KMAX is not read in that file
 * at all and moving it changes nothing here. They are set together on the
 * BOARD because tensor_grouped() requires kgroup == kmax there, and that is a
 * fact about the device path, not about this instrument.
 *
 * CHARSIU_NPU=1 on a machine with no /dev/accel reaches llama_auto_kmax,
 * fails to open, says so, and gives the board's group for free -- bit
 * identical to setting it by hand. It is a good diagnostic and a BAD recipe:
 * on a machine that does have the device it silently becomes a hardware run.
 *
 * The number is exp of the mean negative log likelihood of each token given
 * everything before it, over the token loop, one position at a time. No
 * batching: the batched path is a different arithmetic and this is meant to
 * price the WEIGHTS.
 *
 * The first token has no prediction to score and is skipped. A run reports
 * how many positions it actually scored, because a comparison between two
 * arms is only a comparison if both scored the same ones.
 *
 * --batch SCORES THE OTHER ARITHMETIC, WHICH IS THE ONE THAT SHIPS.
 *
 * The loop above is one position at a time on purpose: it prices the WEIGHTS.
 * But a prompt does not run that way -- llama_prefill_batch takes it in chunks
 * with m > 1, which is a different matmul on different hardware registers, and
 * PLAN.md's whole int4-against-int8 recommendation is about that path. Its
 * quality had never been measured.
 *
 * --batch runs the same corpus through llama_verify_batch, the all-rows form
 * of the batched prefill that speculative decoding already uses, in chunks of
 * llama_prefill_chunk_cap(). Row r of a chunk starting at a is the model's
 * logits at position a + r, so it predicts token a + r + 1 -- exactly the
 * positions the token loop scores, in exactly the same order. The two numbers
 * are therefore comparable, and any gap between them belongs to the batched
 * path and to nothing else.
 *
 * Boundaries checked rather than reasoned about, host, qwen3, chunk cap 160 --
 * n at cap-1, cap, cap+1, cap+2, two chunks, and a corpus that runs out mid
 * chunk. Both modes score the same count and the same ppl at every one:
 *
 *   n = 159  158 positions  45.0441      n = 162  161 positions  44.3203
 *   n = 160  159 positions  45.3804      n = 320  235 positions  45.0442
 *   n = 161  160 positions  44.9461      n = 321  235 positions  45.0442
 *
 * And the refusal fires: CHARSIU_KV_POSMAJOR=1 makes llama_batch_why_not
 * return "a position major KV cache", and --batch then exits 1 with that
 * phrase instead of falling back and printing the token loop's number under
 * this label. The same run without --batch is unaffected.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charsiu_llm.h"

int main(int argc, char **argv)
{
	/*
	 * BEFORE ANY POSITIONAL ARGUMENT IS READ. Several of these tools take
	 * argv[1] straight through atoi, so an unrecognised --version becomes a
	 * dimension of ZERO submitted to the hardware. It also has to exist at
	 * all: tests/board_clk.sh's charsiu_build prints "binary predates the
	 * stamp" for a tool that cannot answer, which is FALSE for these -- they
	 * carry the define and merely had no flag.
	 */
	if (argc > 1 && !strcmp(argv[1], "--version")) {
		printf("%s\n", CHARSIU_BUILD);
		return 0;
	}
	struct llama_model m;
	struct llama_state *st;
	struct tokenizer *tk;
	int32_t *ids = NULL;
	char *text = NULL;
	size_t tlen = 0, cap = 0;
	int n_ctx = 512, want = 0, n = 0, i, scored = 0, batch = 0, top1 = 0;
	double nll = 0.0;
	const char *path = NULL, *tfile = NULL;
	FILE *f;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-c") && i + 1 < argc)
			n_ctx = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-n") && i + 1 < argc)
			want = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--batch"))
			batch = 1;
		/* THE ARGMAX IS ALREADY BEING COMPUTED HERE. The log softmax
		 * needs the largest logit, so the position it came from is one
		 * extra assignment, and printing it turns this into a top-1
		 * oracle for any runtime that can only be asked for a token.
		 * The vendor's cannot be asked for logits at all
		 * (RKLLM_INFER_GET_LOGITS is refused by their export), so
		 * top-1 agreement against a common origin is the only shape of
		 * accuracy comparison their runtime admits. */
		else if (!strcmp(argv[i], "--top1"))
			top1 = 1;
		else if (!path)
			path = argv[i];
		else
			tfile = argv[i];
	}
	if (!path || !tfile) {
		fprintf(stderr, "usage: charsiu_ppl MODEL.gguf TEXT [-c ctx]"
			" [-n tokens] [--batch] [--top1]\n"
			"  --top1  print \"prefix_len argmax_id actual_id\" a line,"
			" so a runtime that\n"
			"          cannot be asked for logits can be scored against"
			" the same origin\n");
		return 2;
	}
	f = fopen(tfile, "rb");
	if (!f) { perror(tfile); return 1; }
	while (1) {
		if (tlen + 65536 + 1 > cap) {
			cap = cap ? cap * 2 : 262144;
			text = realloc(text, cap);
			if (!text) { fprintf(stderr, "oom\n"); return 1; }
		}
		size_t got = fread(text + tlen, 1, 65536, f);
		tlen += got;
		if (got < 65536) break;
	}
	fclose(f);
	text[tlen] = 0;

	if (llama_load(&m, path) < 0) { fprintf(stderr, "load failed\n"); return 1; }
	tk = tokenizer_from_gguf(&m.gguf);
	if (!tk) { fprintf(stderr, "no tokenizer\n"); return 1; }
	ids = malloc((tlen + 8) * sizeof(*ids));
	n = tokenizer_encode(tk, text, 1, ids, (int)(tlen + 8));
	if (n < 2) { fprintf(stderr, "text too short: %d tokens\n", n); return 1; }
	if (want && n > want) n = want;
	if (n > n_ctx) n = n_ctx;

	st = llama_state_new(&m, n_ctx);
	if (!st) { fprintf(stderr, "state failed\n"); return 1; }

	if (batch) {
		/*
		 * REFUSE LOUDLY. A model the batched path declines would
		 * otherwise fall back row by row and produce the token loop's
		 * number under the --batch label, which is the two arms being
		 * one arm.
		 */
		const char *why = llama_batch_why_not(&m);
		int cap = llama_prefill_chunk_cap(&m);
		/*
		 * THE SAME RULE charsiu_run USES, OR THIS MEASURES A WIDTH
		 * NOBODY SHIPS. The runner's default chunk is 80 and
		 * CHARSIU_PREFILL_CHUNK overrides it, capped by the surface
		 * ceiling. This took `cap` -- 160 on qwen3 -- for its first
		 * board round, so the first number ever produced for "the
		 * quality of the batched path" was the quality of a path the
		 * product does not take. Reading the cap is what a probe does
		 * when it forgets the product has a default of its own.
		 *
		 * Honouring the variable also gives the width its own axis, so
		 * a chunk sweep is one env apart from this arm rather than a
		 * rebuild.
		 */
		const char *ec = getenv("CHARSIU_PREFILL_CHUNK");
		int chunk = ec && *ec ? atoi(ec) : 80;
		float *lga;
		int a;

		if (why) {
			fprintf(stderr, "--batch: this model is refused by the "
				"batched path (%s), so there is nothing to "
				"measure. Run without --batch.\n", why);
			return 1;
		}
		if (cap < 2) cap = 2;
		if (chunk > cap) chunk = cap;
		if (chunk < 2) chunk = 2;
		lga = malloc((size_t)chunk * m.n_vocab * sizeof(*lga));
		if (!lga) { fprintf(stderr, "oom on %d x %u logits\n",
				    chunk, m.n_vocab); return 1; }
		for (a = 0; a + 1 < n; a += chunk) {
			int w = n - a < chunk ? n - a : chunk;
			int r;

			if (llama_verify_batch(st, &m, ids + a, w, a, lga)) {
				fprintf(stderr, "batched pass failed at %d\n", a);
				return 1;
			}
			for (r = 0; r < w && a + r + 1 < n; r++) {
				const float *lg = lga + (size_t)r * m.n_vocab;
				float mx = lg[0], sum = 0.0f;
				uint32_t j;

				for (j = 1; j < m.n_vocab; j++)
					if (lg[j] > mx) mx = lg[j];
				for (j = 0; j < m.n_vocab; j++)
					sum += expf(lg[j] - mx);
				nll -= (double)(lg[ids[a + r + 1]] - mx)
				       - log((double)sum);
				scored++;
			}
			fprintf(stderr, "\r  %d/%d  ppl %.4f", scored, n - 1,
				exp(nll / scored));
		}
		free(lga);
		fprintf(stderr, "\r%*s\r", 40, "");
		printf("ppl %.4f  over %d scored positions of %d tokens"
		       "  (%s, --batch in chunks of %d, ceiling %d)\n",
		       exp(nll / scored), scored, n, path, chunk, cap);
		llama_state_free(st);
		tokenizer_free(tk);
		llama_free(&m);
		free(ids); free(text);
		return 0;
	}

	for (i = 0; i + 1 < n; i++) {
		const float *lg = llama_forward(st, ids[i], i);
		float mx = lg[0], sum = 0.0f;
		uint32_t j, am = 0;

		if (!lg) { fprintf(stderr, "forward failed at %d\n", i); return 1; }
		/* log softmax, shifted by the max so the exp cannot overflow */
		for (j = 1; j < m.n_vocab; j++)
			if (lg[j] > mx) { mx = lg[j]; am = j; }
		for (j = 0; j < m.n_vocab; j++)
			sum += expf(lg[j] - mx);
		nll -= (double)(lg[ids[i + 1]] - mx) - log((double)sum);
		/* the prefix length, not i: position i has seen i+1 tokens,
		 * which is what another runtime has to be fed to be asked the
		 * same question */
		if (top1)
			printf("%d %u %u\n", i + 1, am, ids[i + 1]);
		scored++;
		if ((i & 31) == 31)
			fprintf(stderr, "\r  %d/%d  ppl %.4f", scored, n - 1,
				exp(nll / scored));
	}
	fprintf(stderr, "\r%*s\r", 40, "");
	printf("ppl %.4f  over %d scored positions of %d tokens  (%s)\n",
	       exp(nll / scored), scored, n, path);
	llama_state_free(st);
	tokenizer_free(tk);
	llama_free(&m);
	free(ids); free(text);
	return 0;
}
