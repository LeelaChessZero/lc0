/* wrap.c */

/*
This Software is distributed with the following X11 License,
sometimes also known as MIT license.
 
Copyright (c) 2010 Miguel A. Ballicora

 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
*/

#include "wrap.h"

#define LZMA86 
#define ZLIB
#define HUFFMAN
#define LIBLZF
/*#define LIBBZIP2*/

#if defined(LZMA86)
#include "Lzma86Enc.h"
#include "Lzma86Dec.h"
#endif

#if defined(ZLIB)
#include "zlib.h"
#endif

#if defined(HUFFMAN)
#include "hzip.h"
#endif

#if defined(LIBLZF)
#include "lzf.h"
#endif

#if defined(LIBBZIP2)
#include "bzlib.h"
#endif

#if !defined(NDEBUG)
#define NDEBUG
#endif
#ifdef DEBUG
#undef NDEBUG
#endif
#include "assert.h"

/* external, so the compiler can be silenced */
size_t TB_DUMMY_unused;

/***********************************************************************************************************/

extern int
zlib_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	enum COMPRESSION_LEVELS {ZLIB_MAXIMUM_COMPRESSION = 9};
	int outcome;
	unsigned long zz = (unsigned long)out_max;
	outcome = compress2 (out_start, &zz, in_start, (uLong)in_len, ZLIB_MAXIMUM_COMPRESSION);
	*pout_len = (size_t) zz;
	return outcome == Z_OK;
}

extern int
zlib_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	int outcome;
	unsigned long nn = (unsigned long) out_max /* *pout_len */;
	outcome = uncompress (out_start, &nn, in_start, (unsigned long)in_len);
	*pout_len = (size_t)nn;
	return outcome == Z_OK;
}

/***********************************************************************************************************/

extern int
lzf_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	size_t x = lzf_compress (in_start, (unsigned)in_len, out_start, (unsigned)(in_len-1) /* ensures best compression */);
	TB_DUMMY_unused = out_max;
	if (x != 0)
		*pout_len = (size_t) x;
	return x != 0;
}

extern int
lzf_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	*pout_len = (size_t)lzf_decompress (in_start, (unsigned)in_len, out_start, (unsigned)out_max);
	return *pout_len != 0;
}

/***********************************************************************************************************/

extern int
lzma_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	int level  =    5; 			/* 5 => default compression level */
	unsigned int memory = 4096;	/* dictionary size */
	int filter = SZ_FILTER_NO; 	/* => 0, use LZMA, do not try to optimize with x86 filter */
	size_t zz  = out_max; 		/* maximum memory allowed, receives back the actual size */
	int x      = Lzma86_Encode(out_start, &zz, in_start, in_len, level, memory, filter);
	*pout_len  = zz;
	return x == 0;
}

extern int
lzma_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
		size_t nn = out_max;
		int x = Lzma86_Decode(out_start, &nn, in_start, &in_len);
		*pout_len = nn;
		return x == SZ_OK;
}

/***********************************************************************************************************/

#if defined (LIBBZIP2)

extern int
bzip2_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	int	blockSize100k =  9;
	int verbosity     =  0;
	int workFactor    = 30;
	size_t destlen    = out_max;

	int x = BZ2_bzBuffToBuffCompress( (char*)out_start, &destlen, (char*)in_start, in_len, 
								blockSize100k, verbosity, workFactor);
	*pout_len = destlen;
	return x == BZ_OK;
}

extern int
bzip2_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	int	small      = 1;
	int verbosity  = 0;
	size_t destlen = n;

	int x = BZ2_bzBuffToBuffDecompress( (char*)out_start, &destlen, (char*)in_start, in_len,
                               small, verbosity);
	*pout_len = destlen;
	return x == BZ_OK;
}

#endif

/***********************************************************************************************************/

extern int
justcopy_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	size_t i;
	const unsigned char *in  = in_start;
		  unsigned char *out = out_start;

	if (in_len > out_max)
		return 0;

	for (i = 0; i < in_len; i++) {
		*out++ = *in++; 
	}
	*pout_len = in_len;
	return 1;
}

extern int
justcopy_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	size_t i;
	const unsigned char *in  = in_start;
		  unsigned char *out = out_start;

	if (in_len > out_max)
		return 0;

	for (i = 0; i < in_len; i++) {
		*out++ = *in++; 
	}
	*pout_len = in_len;
	return 1;
}

/***********************************************************************************************************/

#define RLE_ESC 253
#define RLE_TER 254
#define RLE_MAX 252
#define TRUE 1
#define FALSE 0

extern int
rle_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	const unsigned char *p;	
	const unsigned char *in      = in_start;
	const unsigned char *in_end  = in  + in_len;
	unsigned char       *out     = out_start;
	int ok = TRUE;
	int ch;
	ptrdiff_t out_len;

	while (in < in_end)
	{
		if (*in == RLE_ESC) {

			*out++ = RLE_ESC;
			*out++ = RLE_ESC;
			in++;	 

		} else {

			ch = *in;
	
			if ( (in_end-in) >= 3 /* enough space for a run */
				&& ch == in[1] && ch == in[2] && ch == in[3] /* enough length */) {

				p = in;
				while (p < in_end && *p == ch && (p-in) < RLE_MAX) {
					p++;
				}

				*out++ = RLE_ESC;
				assert (RLE_MAX < 256);
				*out++ = (unsigned char)(p - in);
				*out++ = (unsigned char)ch;
				in = p;

			} else {	
				*out++ = *in++;	
			}
		}
	}

	if (ok) {
		/*	*out++ = RLE_ESC; *out++ = RLE_TER; */
		out_len = out - out_start;
		*pout_len = (size_t)out_len;	
		ok = (size_t)out_len <= out_max;
	}

	return ok;
}

extern int
rle_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	const unsigned char *in  = in_start;
	const unsigned char *in_end  = in  + in_len;
		  unsigned char *out = out_start;
		  unsigned char *out_end = out + *pout_len;
	int ok = TRUE;
	int ch;
	int n;
	ptrdiff_t out_len;

	while (in < in_end)
	{
									if (in  >=  in_end) { ok = FALSE; break;}
									if (out >= out_end) { ok = FALSE; break;}

		if (*in == RLE_ESC) {
			++in;					if (in >= in_end) { ok = FALSE;	break;}

			if (*in == RLE_ESC) {
				*out++ = *in++;
			} /*else if (*in == RLE_TER) {ok = TRUE;break;} */ else {

				/* rle */
				n  = *in++; 		if (in >= in_end) { ok = FALSE;	break;}
				ch = *in++;			
				while (n-->0) {		if (out >= out_end) { ok = FALSE; break;}
					*out++ = (unsigned char)ch;	
				}
			}
		} else {
			*out++ = *in++;	
		}
	}

	out_len = out - out_start;

	if (ok)
		*pout_len = (size_t)out_len;

	ok = ok && (out_max >= (size_t)out_len);

	return ok;
}
















