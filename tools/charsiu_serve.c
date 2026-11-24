/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * charsiu serve: an OpenAI compatible endpoint, so that every chat front end
 * that already exists works against this board.
 *
 * WHY THIS AND NOT A GUI OF OUR OWN. The audience for this board is not only
 * people who live in a terminal. The shortest path to "open a browser and talk
 * to it" is to speak a protocol the browsers' front ends already speak, which
 * is what ollama did and why it had a graphical story long before it had a GUI.
 * Two endpoints are enough for that:
 *
 *     POST /v1/chat/completions      with stream:true as server-sent events
 *     GET  /v1/models
 *
 * ONE REQUEST AT A TIME, ON PURPOSE. The NPU is a single serial resource:
 * two decodes do not overlap, they queue, and a threaded server would only move
 * the queue somewhere less visible. Requests are served in order.
 *
 * AND THE MODEL STAYS STAGED. Building the NPU tensors takes about twenty
 * seconds, which is the whole reason a server is worth having: it is paid once
 * at startup and never again.
 *
 * WHAT THIS COSTS, SAID PLAINLY: the OpenAI API is stateless, so every
 * request carries the whole conversation and the whole conversation is fed
 * again. On this board a prompt token costs about what a generated one does, so
 * a long history is a real wait before the first new word. That is the contract,
 * not a defect, and a prefix cache is the fix if it ever matters.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define SOCK_BAD INVALID_SOCKET
#define sock_close closesocket
#define SEND_FLAGS 0
#define strncasecmp _strnicmp
#else
#include <unistd.h>
#include <strings.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
typedef int sock_t;
#define SOCK_BAD (-1)
#define sock_close close
#define SEND_FLAGS MSG_NOSIGNAL
#endif
#include "charsiu_llm.h"

static struct llama_model M;
static struct llama_state *ST;
static const char *MODEL_NAME = "charsiu";
static int N_CTX = 2048, N_THREADS = 4;
static float TEMP = 0.0f, TOP_P = 0.9f;
static enum chat_fmt CHAT = CHAT_LLAMA3;   /* set from the vocabulary at load */

/*
 * WHAT THE KV CACHE ACTUALLY HOLDS, prompt and reply both, so the next
 * request can keep the part that has not changed.
 *
 * The API is stateless and the client resends the whole history every turn, so
 * the previous version threw the cache away and re-ran all of it: turn ten
 * re-prefilled the nine turns before it. That is O(n^2) on exactly the workload
 * where prefill hurts, and prefill cannot be made faster here -- the NPU is
 * already at this board's memory roof, and both batching levers measured dead.
 * So it is made to happen once instead.
 */
static int32_t CACHED[16384];
static int CACHED_N;

/* forward one token and remember that the cache now holds it */
static const float *feed(int32_t tok)
{
	const float *l = llama_forward(ST, tok, ST->pos);

	if (CACHED_N < (int)(sizeof(CACHED) / sizeof(CACHED[0])))
		CACHED[CACHED_N++] = tok;
	return l;
}
static uint64_t SEED = 1234;

/* ---- the connection ------------------------------------------------------
 *
 * A SOCKET IS NOT A FILE ON WINDOWS. This used to be fdopen(fd, "r+") and
 * stdio from there, which is fine on Linux and does not exist for a Winsock
 * SOCKET. So the few things the server does with a connection -- read a line,
 * read n bytes, write -- are written once here over recv() and send(), and
 * both systems take the same path.
 *
 * AND A WRITE THAT FAILS ENDS THE ANSWER. A client that hangs up (a Stop
 * button, a closed tab) used to be written to until max_tokens ran out,
 * holding the NPU for an answer nobody would read. `err` is set by the first
 * failed send and the generation loop stops on it.
 */
struct conn {
	sock_t s;
	int err;
	size_t rpos, rlen, wlen;
	char rbuf[8192];
	char wbuf[16384];
};

static int c_fill(struct conn *c)
{
	int n = (int)recv(c->s, c->rbuf, (int)sizeof(c->rbuf), 0);

	if (n <= 0) return 0;
	c->rpos = 0;
	c->rlen = (size_t)n;
	return 1;
}

/* fgets over the connection: up to and including the newline */
static char *c_gets(char *dst, size_t max, struct conn *c)
{
	size_t n = 0;

	if (max < 2) return NULL;
	while (n + 1 < max) {
		if (c->rpos >= c->rlen && !c_fill(c)) break;
		char ch = c->rbuf[c->rpos++];
		dst[n++] = ch;
		if (ch == '\n') break;
	}
	dst[n] = 0;
	return n ? dst : NULL;
}

/* fread over the connection: what has arrived of n bytes */
static size_t c_read(void *dst, size_t n, struct conn *c)
{
	size_t got = 0;

	while (got < n) {
		if (c->rpos >= c->rlen && !c_fill(c)) break;
		size_t k = c->rlen - c->rpos;
		if (k > n - got) k = n - got;
		memcpy((char *)dst + got, c->rbuf + c->rpos, k);
		c->rpos += k;
		got += k;
	}
	return got;
}

static void c_flush(struct conn *c)
{
	size_t off = 0;

	while (!c->err && off < c->wlen) {
		int n = (int)send(c->s, c->wbuf + off, (int)(c->wlen - off), SEND_FLAGS);

		if (n <= 0) { c->err = 1; break; }
		off += (size_t)n;
	}
	c->wlen = 0;
}

static void c_write(struct conn *c, const char *s, size_t n)
{
	while (n) {
		size_t k = sizeof(c->wbuf) - c->wlen;

		if (!k) { c_flush(c); continue; }
		if (k > n) k = n;
		memcpy(c->wbuf + c->wlen, s, k);
		c->wlen += k;
		s += k;
		n -= k;
	}
}

static void c_puts(const char *s, struct conn *c) { c_write(c, s, strlen(s)); }
static void c_putc(int ch, struct conn *c) { char b = (char)ch; c_write(c, &b, 1); }

static void c_printf(struct conn *c, const char *fmt, ...)
{
	char buf[4096];
	va_list ap;
	int n;

	va_start(ap, fmt);
	n = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n < 0) return;
	if ((size_t)n < sizeof(buf)) { c_write(c, buf, (size_t)n); return; }
	/* longer than the stack buffer: format it again into one that fits */
	char *big = malloc((size_t)n + 1);
	if (!big) { c->err = 1; return; }
	va_start(ap, fmt);
	vsnprintf(big, (size_t)n + 1, fmt, ap);
	va_end(ap);
	c_write(c, big, (size_t)n);
	free(big);
}

/* has the other end hung up? Only asked between prompt tokens, before the
 * answer has started and there is nothing to write that would tell. */
static int c_gone(struct conn *c)
{
	fd_set r;
	struct timeval tv = { 0, 0 };
	char b;

	FD_ZERO(&r);
	FD_SET(c->s, &r);
	if (select((int)c->s + 1, &r, NULL, NULL, &tv) <= 0) return 0;
	return recv(c->s, &b, 1, MSG_PEEK) <= 0;
}

/* ---- just enough JSON ---------------------------------------------------- */

/* Find "key" at the top level of an object and return the byte after its colon. */
static const char *j_find(const char *s, const char *end, const char *key)
{
	size_t kl = strlen(key);
	int depth = 0, instr = 0;

	for (const char *p = s; p < end; p++) {
		if (instr) {
			if (*p == '\\') { p++; continue; }
			if (*p == '"') instr = 0;
			continue;
		}
		if (*p == '"') {
			/* only a key at OUR depth counts: a "content" inside a
			 * nested message object must not answer for the request's
			 * own fields. */
			if (depth == 1 && (size_t)(end - p) > kl + 1 &&
			    !strncmp(p + 1, key, kl) && p[kl + 1] == '"') {
				const char *q = p + kl + 2;
				while (q < end && (*q == ' ' || *q == ':')) q++;
				return q;
			}
			instr = 1;
			continue;
		}
		if (*p == '{' || *p == '[') depth++;
		else if (*p == '}' || *p == ']') depth--;
	}
	return NULL;
}

/* Copy a JSON string value into dst, resolving escapes. Returns bytes written. */
static size_t j_str(const char *p, const char *end, char *dst, size_t max)
{
	size_t n = 0;

	if (p >= end || *p != '"') return 0;
	p++;
	while (p < end && *p != '"' && n + 4 < max) {
		if (*p == '\\' && p + 1 < end) {
			p++;
			switch (*p) {
			case 'n': dst[n++] = '\n'; break;
			case 't': dst[n++] = '\t'; break;
			case 'r': dst[n++] = '\r'; break;
			case 'b': dst[n++] = '\b'; break;
			case 'f': dst[n++] = '\f'; break;
			case 'u': {
				/* \uXXXX has to become utf-8 or the prompt is
				 * mangled for every language that needs it. */
				unsigned cp = 0;
				for (int i = 1; i <= 4 && p + i < end; i++) {
					char c = p[i];
					cp = cp * 16 + (c <= '9' ? c - '0'
						      : (c | 32) - 'a' + 10);
				}
				p += 4;
				if (cp < 0x80) dst[n++] = (char)cp;
				else if (cp < 0x800) {
					dst[n++] = (char)(0xc0 | (cp >> 6));
					dst[n++] = (char)(0x80 | (cp & 0x3f));
				} else {
					dst[n++] = (char)(0xe0 | (cp >> 12));
					dst[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
					dst[n++] = (char)(0x80 | (cp & 0x3f));
				}
				break;
			}
			default: dst[n++] = *p; break;
			}
			p++;
			continue;
		}
		dst[n++] = *p++;
	}
	dst[n] = 0;
	return n;
}

/*
 * Copy a message's content.
 *
 * OPENAI SENDS A STRING, ANTHROPIC SENDS EITHER. Its content is often an
 * array of typed blocks, and reading that with j_str() returns nothing at all,
 * so the turn arrives empty and the model answers a question nobody asked.
 * Take the text out of every text block and ignore the rest: an image or a
 * tool_result is not something this can run.
 */
static size_t j_content(const char *p, const char *end, char *dst, size_t max)
{
	int depth = 0, instr = 0;
	const char *obj = NULL;
	size_t n = 0;

	if (p >= end) return 0;
	if (*p == '"') return j_str(p, end, dst, max);
	if (*p != '[') return 0;

	for (p++; p < end; p++) {
		if (instr) {
			if (*p == '\\') { p++; continue; }
			if (*p == '"') instr = 0;
			continue;
		}
		if (*p == '"') { instr = 1; continue; }
		if (*p == '{') { if (depth++ == 0) obj = p; continue; }
		if (*p == '}') {
			if (--depth == 0 && obj) {
				const char *t = j_find(obj, p + 1, "text");

				if (t && *t == '"' && n + 8 < max)
					n += j_str(t, p + 1, dst + n, max - n);
			}
			continue;
		}
		if (*p == ']' && depth == 0) break;
	}
	dst[n] = 0;
	return n;
}

static void j_escape(struct conn *f, const char *s, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];

		switch (c) {
		case '"':  c_puts("\\\"", f); break;
		case '\\': c_puts("\\\\", f); break;
		case '\n': c_puts("\\n", f); break;
		case '\r': c_puts("\\r", f); break;
		case '\t': c_puts("\\t", f); break;
		default:
			/* control bytes must be escaped or the JSON is invalid
			 * and the client drops the whole message. Everything at
			 * 0x20 and above, utf-8 included, goes through as it is. */
			if (c < 0x20) c_printf(f, "\\u%04x", c);
			else c_putc(c, f);
		}
	}
}

/* ---- the prompt ---------------------------------------------------------- */

/*
 * Turn the messages array into a Llama 3 conversation.
 *
 * THE MIDDLE TOKEN OF A TURN HEADER IS ORDINARY TEXT.
 * <|start_header_id|>user<|end_header_id|> is marker, word, marker, and getting
 * that wrong is what put a bare "assistant" into four rounds of board logs.
 */
static size_t build_prompt(const char *body, const char *end, char *out, size_t max)
{
	const char *msgs = j_find(body, end, "messages");
	size_t n = 0;

	if (!msgs || *msgs != '[') return 0;

	/*
	 * ANTHROPIC PUTS THE SYSTEM PROMPT AT THE TOP LEVEL, not in the
	 * messages array. Dropping it means ignoring every instruction a client
	 * like Claude Code sends, while looking like it worked.
	 */
	const char *sys = j_find(body, end, "system");
	if (sys) {
		static char sbuf[16384];

		if (j_content(sys, end, sbuf, sizeof(sbuf)) > 0)
			n += chat_turn(out + n, max - n, CHAT, "system", sbuf);
	}

	const char *p = msgs + 1;
	int depth = 0, instr = 0;
	const char *obj = NULL;

	for (; p < end; p++) {
		if (instr) {
			if (*p == '\\') { p++; continue; }
			if (*p == '"') instr = 0;
			continue;
		}
		if (*p == '"') { instr = 1; continue; }
		if (*p == '{') { if (depth++ == 0) obj = p; continue; }
		if (*p == '}') {
			if (--depth == 0 && obj) {
				char role[32] = "user";
				static char content[32768];
				const char *r = j_find(obj, p + 1, "role");
				const char *c = j_find(obj, p + 1, "content");

				if (r) j_str(r, p + 1, role, sizeof(role));
				content[0] = 0;
				if (c) j_content(c, p + 1, content, sizeof(content));
				n += chat_turn(out + n, max - n, CHAT, role, content);
				if (n + 256 >= max) break;
			}
			continue;
		}
		if (*p == ']' && depth == 0) break;
	}
	n += chat_open(out + n, max - n, CHAT, "assistant");
	return n;
}

/* ---- http ---------------------------------------------------------------- */

static void hdr(struct conn *f, const char *ctype, int chunked)
{
	c_printf(f, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\n", ctype);
	/* so a page served from anywhere can talk to the board directly */
	c_puts("Access-Control-Allow-Origin: *\r\n"
	      "Access-Control-Allow-Headers: *\r\n"
	      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n", f);
	if (chunked) c_puts("Cache-Control: no-cache\r\nConnection: close\r\n", f);
	c_puts("\r\n", f);
}

static void send_models(struct conn *f)
{
	hdr(f, "application/json", 0);
	c_printf(f, "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
		   "\"object\":\"model\",\"created\":%ld,\"owned_by\":\"charsiu\"}]}\n",
		MODEL_NAME, (long)time(NULL));
}

/*
 * One generation loop, two dialects. `anth` selects Anthropic's /v1/messages
 * shape over OpenAI's /v1/chat/completions; everything between the two is the
 * wrapping, so duplicating the sampling loop to serve both would be two places
 * to get temperature and EOG wrong instead of one.
 */
static void send_health(struct conn *f)
{
	hdr(f, "application/json", 0);
	c_printf(f, "{\"status\":\"ok\",\"model\":\"%s\",\"context\":%d}\n",
		MODEL_NAME, N_CTX);
}

/* Anthropic clients ask how big a conversation is before sending it. */
static void count_tokens(struct conn *f, const char *body, size_t blen)
{
	static char prompt[65536];
	static int32_t ids[8192];
	size_t plen = build_prompt(body, body + blen, prompt, sizeof(prompt));
	int n = 0;

	if (plen)
		n = tokenizer_encode(M.tk, prompt, 1, ids,
				     (int)(sizeof(ids) / sizeof(ids[0])));
	if (n < 0) n = 0;
	hdr(f, "application/json", 0);
	c_printf(f, "{\"input_tokens\":%d}\n", n);
}

static void chat(struct conn *f, const char *body, size_t blen, int anth)
{
	const char *end = body + blen;
	static char prompt[65536];
	static int32_t ids[8192];
	int n_gen = 512, stream = 0, i, produced = 0;
	long created = (long)time(NULL);

	const char *v = j_find(body, end, "stream");
	if (v && !strncmp(v, "true", 4)) stream = 1;
	v = j_find(body, end, "max_tokens");
	if (!v) v = j_find(body, end, "max_completion_tokens");
	if (v && *v >= '0' && *v <= '9') n_gen = atoi(v);
	if (n_gen < 1) n_gen = 1;
	if (n_gen > 4096) n_gen = 4096;
	float temp = TEMP;
	v = j_find(body, end, "temperature");
	if (v && ((*v >= '0' && *v <= '9') || *v == '-')) temp = (float)atof(v);

	size_t plen = build_prompt(body, end, prompt, sizeof(prompt));
	if (!plen) {
		c_puts("HTTP/1.1 400 Bad Request\r\n"
		      "Access-Control-Allow-Origin: *\r\n"
		      "Content-Type: application/json\r\n\r\n", f);
		c_puts(anth ? "{\"type\":\"error\",\"error\":{\"type\":"
			     "\"invalid_request_error\",\"message\":\"no messages\"}}\n"
			   : "{\"error\":{\"message\":\"no messages\"}}\n", f);
		return;
	}

	if (!ST) {
		ST = llama_state_new(&M, N_CTX);
		CACHED_N = 0;
	}
	if (!ST) { c_puts("HTTP/1.1 500 Internal Server Error\r\n\r\n", f); return; }
	charsiu_npu_idle(ST->pool.dev, 0);   /* the hold is back before the first token */

	int n_ids = tokenizer_encode(M.tk, prompt, 1, ids, (int)(sizeof(ids)/sizeof(ids[0])));
	if (n_ids < 1 || n_ids + 8 >= ST->n_ctx) {
		c_printf(f, "HTTP/1.1 400 Bad Request\r\n"
			   "Access-Control-Allow-Origin: *\r\n"
			   "Content-Type: application/json\r\n\r\n");
		c_printf(f, anth ? "{\"type\":\"error\",\"error\":{\"type\":"
				  "\"invalid_request_error\",\"message\":\"the "
				  "conversation does not fit in %d tokens\"}}\n"
				: "{\"error\":{\"message\":\"the conversation "
				  "does not fit in %d tokens\"}}\n", ST->n_ctx);
		return;
	}
	if (n_ids + n_gen >= ST->n_ctx) n_gen = ST->n_ctx - n_ids - 1;

	/*
	 * COMPARE THE TOKENS, DO NOT TRUST THE HISTORY. The client may edit,
	 * branch or truncate what it sends, so the only safe thing is the
	 * longest prefix that is IDENTICAL token for token. Anything after the
	 * first difference is recomputed, which is the same answer the old
	 * code gave -- it just no longer recomputes the part that matched.
	 *
	 * AND AT LEAST ONE TOKEN MUST BE FED, or there are no logits to
	 * sample from. Keeping all n would return whatever the last request
	 * left in the buffer.
	 */
	int keep = 0;
	/* the control, so a round can measure against the old behaviour
	 * without rebuilding: CHARSIU_NO_KV_REUSE throws the cache away the
	 * way this used to. */
	static int noreuse = -1;

	if (noreuse < 0)
		noreuse = charsiu_env_flag("CHARSIU_NO_KV_REUSE", 0);
	while (!noreuse && keep < CACHED_N && keep < n_ids - 1 &&
	       CACHED[keep] == ids[keep])
		keep++;
	if (keep > ST->pos)
		keep = ST->pos;          /* never claim more than the cache holds */
	ST->pos = keep;
	CACHED_N = keep;

	const float *logits = NULL;
	for (i = keep; i < n_ids; i++) {
		logits = feed(ids[i]);
		/* a long prompt is seconds of work before the first byte of
		 * the answer, and a Stop pressed in that time has to count */
		if (((i - keep) & 15) == 15 && c_gone(f)) {
			fprintf(stderr, "  the client went away during the "
					"prompt; stopped\n");
			return;
		}
	}
	if (keep)
		fprintf(stderr, "  kv: kept %d of %d prompt tokens\n", keep, n_ids);

	if (stream) hdr(f, "text/event-stream", 1);
	if (stream && anth) {
		c_printf(f, "event: message_start\ndata: {\"type\":\"message_start\","
			   "\"message\":{\"id\":\"msg_%ld\",\"type\":\"message\","
			   "\"role\":\"assistant\",\"model\":\"%s\",\"content\":[],"
			   "\"stop_reason\":null,\"stop_sequence\":null,"
			   "\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}}\n\n",
			created, MODEL_NAME, n_ids);
		c_puts("event: content_block_start\ndata: {\"type\":"
		      "\"content_block_start\",\"index\":0,\"content_block\":"
		      "{\"type\":\"text\",\"text\":\"\"}}\n\n", f);
		c_flush(f);
	}

	static char acc[65536];
	size_t alen = 0;
	int hit_eog = 0;

	for (i = 0; i < n_gen; i++) {
		int32_t tok = temp > 0.0f
			? llama_sample(logits, M.n_vocab, temp, TOP_P, &SEED)
			: llama_argmax(logits, M.n_vocab);
		int len;
		const char *s;

		if (tokenizer_is_eog(M.tk, tok)) { hit_eog = 1; break; }
		s = tokenizer_decode(M.tk, tok, &len);
		if (!tokenizer_is_control(M.tk, tok) && len > 0) {
			if (stream && anth) {
				c_puts("event: content_block_delta\ndata: {\"type\":"
				      "\"content_block_delta\",\"index\":0,\"delta\":"
				      "{\"type\":\"text_delta\",\"text\":\"", f);
				j_escape(f, s, (size_t)len);
				c_puts("\"}}\n\n", f);
				c_flush(f);
			} else if (stream) {
				c_printf(f, "data: {\"id\":\"chatcmpl-%ld\",\"object\":"
					   "\"chat.completion.chunk\",\"created\":%ld,"
					   "\"model\":\"%s\",\"choices\":[{\"index\":0,"
					   "\"delta\":{\"content\":\"",
					created, created, MODEL_NAME);
				j_escape(f, s, (size_t)len);
				c_puts("\"},\"finish_reason\":null}]}\n\n", f);
				c_flush(f);
			} else if (alen + (size_t)len < sizeof(acc)) {
				memcpy(acc + alen, s, (size_t)len);
				alen += (size_t)len;
			}
		}
		produced++;
		if (f->err) {
			fprintf(stderr, "  the client went away after %d tokens; "
					"stopped\n", produced);
			break;
		}
		if (ST->pos >= ST->n_ctx) break;
		logits = feed(tok);
		if (!logits) break;
	}

	/* Both dialects distinguish "it finished" from "it ran out of room",
	 * and this reported "stop" either way, so a truncated answer looked
	 * complete to any client that checks. */
	const char *why = anth ? (hit_eog ? "end_turn" : "max_tokens")
			       : (hit_eog ? "stop" : "length");

	if (stream && anth) {
		c_puts("event: content_block_stop\ndata: {\"type\":"
		      "\"content_block_stop\",\"index\":0}\n\n", f);
		c_printf(f, "event: message_delta\ndata: {\"type\":\"message_delta\","
			   "\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},"
			   "\"usage\":{\"output_tokens\":%d}}\n\n", why, produced);
		c_puts("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", f);
	} else if (stream) {
		c_printf(f, "data: {\"id\":\"chatcmpl-%ld\",\"object\":"
			   "\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
			   "\"choices\":[{\"index\":0,\"delta\":{},"
			   "\"finish_reason\":\"%s\"}]}\n\n"
			   "data: [DONE]\n\n", created, created, MODEL_NAME, why);
	} else if (anth) {
		hdr(f, "application/json", 0);
		c_printf(f, "{\"id\":\"msg_%ld\",\"type\":\"message\","
			   "\"role\":\"assistant\",\"model\":\"%s\","
			   "\"content\":[{\"type\":\"text\",\"text\":\"",
			created, MODEL_NAME);
		j_escape(f, acc, alen);
		c_printf(f, "\"}],\"stop_reason\":\"%s\",\"stop_sequence\":null,"
			   "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}\n",
			why, n_ids, produced);
	} else {
		hdr(f, "application/json", 0);
		c_printf(f, "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\","
			   "\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,"
			   "\"message\":{\"role\":\"assistant\",\"content\":\"",
			created, created, MODEL_NAME);
		j_escape(f, acc, alen);
		c_printf(f, "\"},\"finish_reason\":\"%s\"}],\"usage\":{"
			   "\"prompt_tokens\":%d,\"completion_tokens\":%d,"
			   "\"total_tokens\":%d}}\n",
			why, n_ids, produced, n_ids + produced);
	}
	c_flush(f);
}

static void serve_one(sock_t fd)
{
	static struct conn cn;
	struct conn *f = &cn;
	char line[8192], method[16] = "", path[256] = "";
	size_t clen = 0;

	memset(f, 0, offsetof(struct conn, rbuf));
	f->s = fd;

	if (!c_gets(line, sizeof(line), f)) { sock_close(fd); return; }
	sscanf(line, "%15s %255s", method, path);
	while (c_gets(line, sizeof(line), f)) {
		if (line[0] == '\r' || line[0] == '\n') break;
		if (!strncasecmp(line, "Content-Length:", 15))
			clen = (size_t)strtoul(line + 15, NULL, 10);
	}

	if (!strcmp(method, "OPTIONS")) {
		hdr(f, "text/plain", 0);
	} else if (!strcmp(method, "GET") && strstr(path, "/v1/models")) {
		send_models(f);
	} else if (!strcmp(method, "GET") &&
		   (!strcmp(path, "/health") || !strcmp(path, "/v1/health"))) {
		/* Nearly every client probes this before it will talk. */
		send_health(f);
	} else if (!strcmp(method, "GET") && !strcmp(path, "/")) {
		hdr(f, "text/plain", 0);
		c_printf(f, "charsiu, model %s, context %d\n"
			   "POST /v1/chat/completions      (OpenAI)\n"
			   "POST /v1/messages              (Anthropic)\n"
			   "POST /v1/messages/count_tokens\n"
			   "GET  /v1/models   GET /health\n",
			MODEL_NAME, N_CTX);
	} else if (!strcmp(method, "POST") &&
		   (strstr(path, "/chat/completions") || strstr(path, "/messages"))) {
		char *body = malloc(clen + 1);

		if (!body) { sock_close(fd); return; }
		size_t got = clen ? c_read(body, clen, f) : 0;
		body[got] = 0;
		fprintf(stderr, "  POST %s  %zu bytes\n", path, got);
		/* count_tokens must be tested BEFORE /messages: it contains it. */
		if (strstr(path, "/count_tokens"))
			count_tokens(f, body, got);
		else
			chat(f, body, got, strstr(path, "/messages") != NULL);
		free(body);
	} else {
		c_puts("HTTP/1.1 404 Not Found\r\n"
		      "Access-Control-Allow-Origin: *\r\n\r\n", f);
	}
	c_flush(f);
	sock_close(fd);
}

int main(int argc, char **argv)
{
	/* before any positional argument is read: several of these tools take
	 * argv[1] straight through atoi, so an unrecognised --version becomes a
	 * dimension of ZERO submitted to the hardware. npu_slice_test did
	 * exactly that until this went in. */
	if (argc > 1 && !strcmp(argv[1], "--version")) {
		printf("%s\n", CHARSIU_BUILD);
		return 0;
	}
	const char *path = NULL;
	int port = 11434, i;   /* ollama's port, so existing clients just work */
	/*
	 * WINDOWS LISTENS ON THE LOOPBACK BY DEFAULT. A Windows machine is a
	 * desktop on somebody's network, and binding every interface there both
	 * offers the model to the LAN and stops at a firewall prompt. A board on
	 * Linux is usually reached from another machine, so it keeps 0.0.0.0.
	 */
#ifdef _WIN32
	const char *host = "127.0.0.1";
#else
	const char *host = "0.0.0.0";
#endif
	const char *cache = NULL;
	static char cachepath[1024];

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
		else if (!strcmp(argv[i], "-c") && i + 1 < argc) N_CTX = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) N_THREADS = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--name") && i + 1 < argc) MODEL_NAME = argv[++i];
		else if (!strcmp(argv[i], "--cache")) cache = "";
		else if (!strcmp(argv[i], "--cache-at") && i + 1 < argc) cache = argv[++i];
		else if (argv[i][0] != '-' && !path) path = argv[i];
		else {
			fprintf(stderr,
"usage: charsiu_serve MODEL.gguf [--port N] [--host A] [-c CTX] [-t THREADS] [--name NAME]\n"
"                    [--cache | --cache-at FILE]\n"
"  --port 0 lets the system choose; the ready line names the port it got\n"
"  --cache keeps the quantised weights in MODEL.gguf.wcache, as charsiu_run's\n"
"  POST /v1/chat/completions   (stream:true supported)\n"
"  GET  /v1/models\n");
			return 2;
		}
	}
	if (!path) { fprintf(stderr, "charsiu_serve: which model?\n"); return 2; }

#ifdef _WIN32
	{
		WSADATA wd;

		if (WSAStartup(MAKEWORD(2, 2), &wd)) {
			fprintf(stderr, "charsiu_serve: Winsock would not start\n");
			return 1;
		}
	}
#else
	/* a client that hangs up mid-stream must not take the server with it */
	signal(SIGPIPE, SIG_IGN);
#endif

	/*
	 * THE SAME WEIGHT CACHE charsiu_run --cache HAS, and the same stamp. On
	 * Windows it is the difference between a minute of quantising at every
	 * start and five seconds of reading (docs/M7.md in woa-rk3576-npu).
	 */
	if (cache) {
		struct stat sb;
		char stamp[64];

		if (!*cache) {
			snprintf(cachepath, sizeof(cachepath), "%s.wcache", path);
			cache = cachepath;
		}
		if (!stat(path, &sb))
			snprintf(stamp, sizeof(stamp), "%llu:%llu",
				 (unsigned long long)sb.st_size,
				 (unsigned long long)sb.st_mtime);
		else
			snprintf(stamp, sizeof(stamp), "nostat");
		setenv("CHARSIU_NPU_CACHE", cache, 1);
		setenv("CHARSIU_NPU_CACHE_STAMP", stamp, 1);
		fprintf(stderr, "  weight cache: %s\n", cache);
	}

	fprintf(stderr, "charsiu_serve: loading %s\n", path);
	if (llama_load(&M, path) < 0) return 1;
	CHAT = chat_format_of(M.tk);
	fprintf(stderr, "  chat format: %s\n",
		CHAT == CHAT_PHI3  ? "phi3"  :
		CHAT == CHAT_GEMMA ? "gemma" :
		CHAT == CHAT_GEMMA4 ? "gemma4" :
		CHAT == CHAT_CHATML ? "chatml" : "llama3");
	if (!MODEL_NAME[0] || !strcmp(MODEL_NAME, "charsiu")) {
		const char *b = strrchr(path, '/');
		const char *b2 = strrchr(path, '\\');

		if (b2 && (!b || b2 > b)) b = b2;
		MODEL_NAME = b ? b + 1 : path;
	}
	if (N_CTX <= 0) N_CTX = (int)M.n_ctx_train;
	ST = llama_state_new(&M, N_CTX);
	if (!ST) { fprintf(stderr, "charsiu_serve: no room for %d tokens\n", N_CTX); return 1; }

	/* stage the NPU NOW, not on the first request. Twenty seconds of
	 * silence is acceptable at startup and is not acceptable in a reply. */
	{
		int32_t warm[4];
		int n = tokenizer_encode(M.tk, "hello", 1, warm, 4);

		for (i = 0; i < n; i++) llama_forward(ST, warm[i], ST->pos);
		llama_state_free(ST);
		ST = llama_state_new(&M, N_CTX);
	}
	/*
	 * THE QoS HOLD IS FOR A REQUEST, NOT FOR THE PROCESS. Opening the
	 * NPU forbids the CPUs their deep idle states -- a quarter of decode
	 * on this board -- and a one-shot run closes the device minutes
	 * later. A server sits at accept() for hours between requests, and
	 * an idle board that cannot idle is a heater. Dropped here and around
	 * every request below; the hold is back before the first token.
	 */
	if (ST) charsiu_npu_idle(ST->pool.dev, 1);

	sock_t s = socket(AF_INET, SOCK_STREAM, 0);
	int one = 1;
	struct sockaddr_in a;
	socklen_t alen = sizeof(a);

	if (s == SOCK_BAD) {
		fprintf(stderr, "charsiu_serve: no socket\n");
		return 1;
	}
#ifndef _WIN32
	/* on Windows SO_REUSEADDR lets a second server take a port that is in
	 * use, which is the opposite of what it means here */
	setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
#else
	setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *)&one, sizeof(one));
#endif
	memset(&a, 0, sizeof(a));
	a.sin_family = AF_INET;
	a.sin_port = htons((unsigned short)port);
	if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
		fprintf(stderr, "charsiu_serve: --host %s is not an IPv4 address\n", host);
		return 2;
	}
	if (bind(s, (struct sockaddr *)&a, sizeof(a)) < 0) {
#ifdef _WIN32
		fprintf(stderr, "charsiu_serve: port %d: Winsock error %d\n", port, WSAGetLastError());
#else
		fprintf(stderr, "charsiu_serve: port %d: %s\n", port, strerror(errno));
#endif
		return 1;
	}
	listen(s, 8);
	/* THE PORT IT GOT, not the one it was asked for: with --port 0 the
	 * system chooses, and whoever started this reads it from here. */
	if (!getsockname(s, (struct sockaddr *)&a, &alen))
		port = ntohs(a.sin_port);
	fprintf(stderr, "charsiu_serve: %s on http://%s:%d  (model %s, ctx %d)\n"
			"  one request at a time: the NPU is serial, so they queue\n",
		"ready", host, port, MODEL_NAME, N_CTX);

	for (;;) {
		sock_t c = accept(s, NULL, NULL);

		if (c == SOCK_BAD) {
#ifndef _WIN32
			if (errno == EINTR) continue;
#endif
			break;
		}
		serve_one(c);
		/* the request is answered: let the CPUs sleep until the next */
		if (ST) charsiu_npu_idle(ST->pool.dev, 1);
	}
	return 0;
}
