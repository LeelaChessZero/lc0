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

/*--------------------------------------------------------------------------*\
|
|                   Compressing wrapper functions
|
*---------------------------------------------------------------------------*/
#define MAXBLOCK (1 << 16)

#include <stdlib.h>
#include "gtb-dec.h"
#include "hzip.h"
#include "wrap.h"

typedef int bool_t;

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0;
#endif

/*static unsigned char intermediate_block[MAXBLOCK] ;*/

static int DECODE_SCHEME = CP4;
static int CP_SCHEME = CP4;

static int f_decode (size_t z, unsigned char *bz, size_t n, unsigned char *bp);

extern void
set_decoding_scheme(int x)
{
	DECODE_SCHEME = x;
	CP_SCHEME     = x;
}

extern int decoding_scheme(void)
{
	return DECODE_SCHEME;
}

extern int 
decode (size_t z, unsigned char *bz, size_t n, unsigned char *bp)
{
	return f_decode (z, bz, n, bp);
}

/*======================== WRAPPERS ========================*/

static int 
f_decode (size_t z, unsigned char *bz, size_t n, unsigned char *bp)
{
/* 	bp buffer provided 
|	bz buffer "zipped", compressed
|	n  len of buffer provided
|	z  len of buffer zipped	
\*---------------------------------------------------------------*/

/*
	unsigned char *ib = intermediate_block;
	unsigned int m;		
	return	huff_decode (bz, z, ib, &m, MAXBLOCK) && rle_decode (ib, m, bp, &n, MAXBLOCK);
*/

	if        (CP_SCHEME == CP1) {

		/* HUFFMAN */
		return huff_decode (bz, z, bp, &n, MAXBLOCK);

	} else if (CP_SCHEME == CP2) {

		/* LZF */
		return lzf_decode  (bz, z, bp, &n, MAXBLOCK);

	} else if (CP_SCHEME == CP3) {

		/* ZLIB */
		return zlib_decode (bz, z, bp, &n, MAXBLOCK);

	} else if (CP_SCHEME == CP4) {

		/* LZMA86 */
		return lzma_decode (bz, z, bp, &n, n); /* maximum needs to be the exact number that it will produce */

	} else if (CP_SCHEME == CP7) {

		/* RLE */
		return rle_decode (bz, z, bp, &n, MAXBLOCK);

	#if defined (LIBBZIP2)
	} else if (CP_SCHEME == CP8) {

		/* BZIP2 */
		return bzip2_decode (bz, z, bp, &n, MAXBLOCK);
	#endif

	} else if (CP_SCHEME == CP9) {

		return justcopy_decode (bz, z, bp, &n, MAXBLOCK);

	} else {

		return FALSE;
	}
}

