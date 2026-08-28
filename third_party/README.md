# third_party

## stb_image.h

v2.30, public domain (or MIT, at your option), from https://github.com/nothings/stb

⚠ **Vendored rather than depended on.** charsiu builds with `-lm -lpthread` and
nothing else, on a board where the distribution is whatever fitted on the card.
A JPEG and PNG decoder is not something to write, and it is not something to
make somebody install either.

Only `stb_image.h` is here. The resize is ours (`src/image.c`): it is forty
lines of bilinear, it has a reference test, and it saves a second seven thousand
line header.

The licence is compatible with this repository's GPL-2.0. It is unmodified, and
it should stay that way: fixes go upstream.
