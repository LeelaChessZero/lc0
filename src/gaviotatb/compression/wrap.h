/* wrap.h */

/*
 X11 License:

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

#if !defined(H_WRAP)
#define H_WRAP

/*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/
#include <stdlib.h>

extern int zlib_encode (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
extern int zlib_decode (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);

extern int lzf_encode  (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
extern int lzf_decode  (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);

extern int lzma_encode (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
extern int lzma_decode (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);

#if defined (LIBBZIP2)
extern int bzip2_encode(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
extern int bzip2_decode(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
#endif

extern int rle_encode  (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
extern int rle_decode  (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);

extern int justcopy_encode (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);
extern int justcopy_decode (const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max);


/*<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/
#endif
