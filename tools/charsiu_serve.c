/*
 * Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
 * SPDX-License-Identifier: GPL-2.0
 *
 * charsiu serve: an OpenAI compatible endpoint, so that every chat front end
 * that already exists works against this board.
 *
 * ⚠ WHY THIS AND NOT A GUI OF OUR OWN. The audience for this board is not only
 * people who live in a terminal. The shortest path to "open a browser and talk
 * to it" is to speak a protocol the browsers' front ends already speak, which
 * is what ollama did and why it had a graphical story long before it had a GUI.
 * Two endpoints are enough for that:
 *
 *     POST /v1/chat/completions      with stream:true as server-sent events
 *     GET  /v1/models
 *
 * ⚠ ONE REQUEST AT A TIME, ON PURPOSE. The NPU is a single serial resource:
 * two decodes do not overlap, they queue, and a threaded server would only move
 * the queue somewhere less visible. Requests are served in order.
 *
 * ⚠ AND THE MODEL STAYS STAGED. Building the NPU tensors takes about twenty
 * seconds, which is the whole reason a server is worth having: it is paid once
 * at startup and never again.
 *
 * ⚠ WHAT THIS COSTS, SAID PLAINLY: the OpenAI API is stateless, so every
 * request carries the whole conversation and the whole conversation is fed
 * again. On this board a prompt token costs about what a generated one does, so
 * a long history is a real wait before the first new word. That is the contract,
 * not a defect, and a prefix cache is the fix if it ever matters.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include "charsiu_llm.h"

static struct llama_model M;
static struct llama_state *ST;
static const char *MODEL_NAME = "charsiu";
static int N_CTX = 2048, N_THREADS = 4;
static float TEMP = 0.0f, TOP_P = 0.9f;
static uint64_t SEED = 1234;

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
			/* ⚠ only a key at OUR depth counts: a "content" inside a
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
				/* ⚠ \uXXXX has to become utf-8 or the prompt is
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

static void j_escape(FILE *f, const char *s, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];

		switch (c) {
		case '"':  fputs("\\\"", f); break;
		case '\\': fputs("\\\\", f); break;
		case '\n': fputs("\\n", f); break;
		case '\r': fputs("\\r", f); break;
		case '\t': fputs("\\t", f); break;
		default:
			/* ⚠ control bytes must be escaped or the JSON is invalid
			 * and the client drops the whole message. Everything at
			 * 0x20 and above, utf-8 included, goes through as it is. */
			if (c < 0x20) fprintf(f, "\\u%04x", c);
			else fputc(c, f);
		}
	}
}

/* ---- the prompt ---------------------------------------------------------- */

/*
 * Turn the messages array into a Llama 3 conversation.
 *
 * ⚠ THE MIDDLE TOKEN OF A TURN HEADER IS ORDINARY TEXT.
 * <|start_header_id|>user<|end_header_id|> is marker, word, marker, and getting
 * that wrong is what put a bare "assistant" into four rounds of board logs.
 */
static size_t build_prompt(const char *body, const char *end, char *out, size_t max)
{
	const char *msgs = j_find(body, end, "messages");
	size_t n = 0;

	if (!msgs || *msgs != '[') return 0;

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
				if (c) j_str(c, p + 1, content, sizeof(content));
				n += (size_t)snprintf(out + n, max - n,
					"<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>",
					role, content);
				if (n + 256 >= max) break;
			}
			continue;
		}
		if (*p == ']' && depth == 0) break;
	}
	n += (size_t)snprintf(out + n, max - n,
			      "<|start_header_id|>assistant<|end_header_id|>\n\n");
	return n;
}

/* ---- http ---------------------------------------------------------------- */

static void hdr(FILE *f, const char *ctype, int chunked)
{
	fprintf(f, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\n", ctype);
	/* so a page served from anywhere can talk to the board directly */
	fputs("Access-Control-Allow-Origin: *\r\n"
	      "Access-Control-Allow-Headers: *\r\n"
	      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n", f);
	if (chunked) fputs("Cache-Control: no-cache\r\nConnection: close\r\n", f);
	fputs("\r\n", f);
}

static void send_models(FILE *f)
{
	hdr(f, "application/json", 0);
	fprintf(f, "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
		   "\"object\":\"model\",\"created\":%ld,\"owned_by\":\"charsiu\"}]}\n",
		MODEL_NAME, (long)time(NULL));
}

static void chat(FILE *f, const char *body, size_t blen)
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
		fputs("HTTP/1.1 400 Bad Request\r\n"
		      "Access-Control-Allow-Origin: *\r\n"
		      "Content-Type: application/json\r\n\r\n"
		      "{\"error\":{\"message\":\"no messages\"}}\n", f);
		return;
	}

	/*
	 * ⚠ A FRESH STATE PER REQUEST. The API is stateless and the client sends
	 * the whole history every time, so keeping a previous kv cache would
	 * make the two disagree about what was said. The MODEL and the staged
	 * NPU tensors are what persist, and they are the expensive part.
	 */
	llama_state_free(ST);
	ST = llama_state_new(&M, N_CTX);
	if (!ST) { fputs("HTTP/1.1 500 Internal Server Error\r\n\r\n", f); return; }

	int n_ids = tokenizer_encode(M.tk, prompt, 1, ids, (int)(sizeof(ids)/sizeof(ids[0])));
	if (n_ids < 1 || n_ids + 8 >= ST->n_ctx) {
		fprintf(f, "HTTP/1.1 400 Bad Request\r\n"
			   "Access-Control-Allow-Origin: *\r\n"
			   "Content-Type: application/json\r\n\r\n"
			   "{\"error\":{\"message\":\"the conversation does not fit in %d tokens\"}}\n",
			ST->n_ctx);
		return;
	}
	if (n_ids + n_gen >= ST->n_ctx) n_gen = ST->n_ctx - n_ids - 1;

	const float *logits = NULL;
	for (i = 0; i < n_ids; i++)
		logits = llama_forward(ST, ids[i], ST->pos);

	if (stream) hdr(f, "text/event-stream", 1);

	static char acc[65536];
	size_t alen = 0;

	for (i = 0; i < n_gen; i++) {
		int32_t tok = temp > 0.0f
			? llama_sample(logits, M.n_vocab, temp, TOP_P, &SEED)
			: llama_argmax(logits, M.n_vocab);
		int len;
		const char *s;

		if (tokenizer_is_eog(M.tk, tok)) break;
		s = tokenizer_decode(M.tk, tok, &len);
		if (!tokenizer_is_control(M.tk, tok) && len > 0) {
			if (stream) {
				fprintf(f, "data: {\"id\":\"chatcmpl-%ld\",\"object\":"
					   "\"chat.completion.chunk\",\"created\":%ld,"
					   "\"model\":\"%s\",\"choices\":[{\"index\":0,"
					   "\"delta\":{\"content\":\"",
					created, created, MODEL_NAME);
				j_escape(f, s, (size_t)len);
				fputs("\"},\"finish_reason\":null}]}\n\n", f);
				fflush(f);
			} else if (alen + (size_t)len < sizeof(acc)) {
				memcpy(acc + alen, s, (size_t)len);
				alen += (size_t)len;
			}
		}
		produced++;
		if (ST->pos >= ST->n_ctx) break;
		logits = llama_forward(ST, tok, ST->pos);
		if (!logits) break;
	}

	if (stream) {
		fprintf(f, "data: {\"id\":\"chatcmpl-%ld\",\"object\":"
			   "\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
			   "\"choices\":[{\"index\":0,\"delta\":{},"
			   "\"finish_reason\":\"stop\"}]}\n\n"
			   "data: [DONE]\n\n", created, created, MODEL_NAME);
	} else {
		hdr(f, "application/json", 0);
		fprintf(f, "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\","
			   "\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,"
			   "\"message\":{\"role\":\"assistant\",\"content\":\"",
			created, created, MODEL_NAME);
		j_escape(f, acc, alen);
		fprintf(f, "\"},\"finish_reason\":\"stop\"}],\"usage\":{"
			   "\"prompt_tokens\":%d,\"completion_tokens\":%d,"
			   "\"total_tokens\":%d}}\n",
			n_ids, produced, n_ids + produced);
	}
	fflush(f);
}

static void serve_one(int fd)
{
	FILE *f = fdopen(fd, "r+");
	char line[8192], method[16] = "", path[256] = "";
	size_t clen = 0;

	if (!f) { close(fd); return; }
	setvbuf(f, NULL, _IONBF, 0);

	if (!fgets(line, sizeof(line), f)) { fclose(f); return; }
	sscanf(line, "%15s %255s", method, path);
	while (fgets(line, sizeof(line), f)) {
		if (line[0] == '\r' || line[0] == '\n') break;
		if (!strncasecmp(line, "Content-Length:", 15))
			clen = (size_t)strtoul(line + 15, NULL, 10);
	}

	if (!strcmp(method, "OPTIONS")) {
		hdr(f, "text/plain", 0);
	} else if (!strcmp(method, "GET") && strstr(path, "/v1/models")) {
		send_models(f);
	} else if (!strcmp(method, "GET") && !strcmp(path, "/")) {
		hdr(f, "text/plain", 0);
		fprintf(f, "charsiu, model %s, context %d\n"
			   "POST /v1/chat/completions   GET /v1/models\n",
			MODEL_NAME, N_CTX);
	} else if (!strcmp(method, "POST") && strstr(path, "/chat/completions")) {
		char *body = malloc(clen + 1);

		if (!body) { fclose(f); return; }
		size_t got = clen ? fread(body, 1, clen, f) : 0;
		body[got] = 0;
		fprintf(stderr, "  POST /v1/chat/completions  %zu bytes\n", got);
		chat(f, body, got);
		free(body);
	} else {
		fputs("HTTP/1.1 404 Not Found\r\n"
		      "Access-Control-Allow-Origin: *\r\n\r\n", f);
	}
	fclose(f);
}

int main(int argc, char **argv)
{
	const char *path = NULL;
	int port = 11434, i;   /* ollama's port, so existing clients just work */
	const char *host = "0.0.0.0";

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
		else if (!strcmp(argv[i], "-c") && i + 1 < argc) N_CTX = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) N_THREADS = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--name") && i + 1 < argc) MODEL_NAME = argv[++i];
		else if (argv[i][0] != '-' && !path) path = argv[i];
		else {
			fprintf(stderr,
"usage: charsiu_serve MODEL.gguf [--port N] [--host A] [-c CTX] [-t THREADS] [--name NAME]\n"
"  POST /v1/chat/completions   (stream:true supported)\n"
"  GET  /v1/models\n");
			return 2;
		}
	}
	if (!path) { fprintf(stderr, "charsiu_serve: which model?\n"); return 2; }

	/* ⚠ a client that hangs up mid-stream must not take the server with it */
	signal(SIGPIPE, SIG_IGN);

	fprintf(stderr, "charsiu_serve: loading %s\n", path);
	if (llama_load(&M, path) < 0) return 1;
	if (!MODEL_NAME[0] || !strcmp(MODEL_NAME, "charsiu")) {
		const char *b = strrchr(path, '/');
		MODEL_NAME = b ? b + 1 : path;
	}
	if (N_CTX <= 0) N_CTX = (int)M.n_ctx_train;
	ST = llama_state_new(&M, N_CTX);
	if (!ST) { fprintf(stderr, "charsiu_serve: no room for %d tokens\n", N_CTX); return 1; }

	/* ⚠ stage the NPU NOW, not on the first request. Twenty seconds of
	 * silence is acceptable at startup and is not acceptable in a reply. */
	{
		int32_t warm[4];
		int n = tokenizer_encode(M.tk, "hello", 1, warm, 4);

		for (i = 0; i < n; i++) llama_forward(ST, warm[i], ST->pos);
		llama_state_free(ST);
		ST = llama_state_new(&M, N_CTX);
	}

	int s = socket(AF_INET, SOCK_STREAM, 0);
	int one = 1;
	struct sockaddr_in a;

	setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	memset(&a, 0, sizeof(a));
	a.sin_family = AF_INET;
	a.sin_port = htons((unsigned short)port);
	a.sin_addr.s_addr = !strcmp(host, "127.0.0.1") ? htonl(INADDR_LOOPBACK)
							: htonl(INADDR_ANY);
	if (bind(s, (struct sockaddr *)&a, sizeof(a)) < 0) {
		fprintf(stderr, "charsiu_serve: port %d: %s\n", port, strerror(errno));
		return 1;
	}
	listen(s, 8);
	fprintf(stderr, "charsiu_serve: %s on http://%s:%d  (model %s, ctx %d)\n"
			"  one request at a time: the NPU is serial, so they queue\n",
		"ready", host, port, MODEL_NAME, N_CTX);

	for (;;) {
		int c = accept(s, NULL, NULL);

		if (c < 0) { if (errno == EINTR) continue; break; }
		serve_one(c);
	}
	return 0;
}
