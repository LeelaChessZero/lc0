/* hzip.c */
/*
|	Routines designed to be used as a pilot experiment for compression
|	of tablebases. Not really optimized, but they are supposed to work
|	--Miguel A. Ballicora
*/

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

#include "hzip.h"

/*-------------------------------------------------------------------*\
|
|	                 Huffman coding compression
|
\*-------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAXDIVERSITY (256)
#define MAXHEAP (MAXDIVERSITY+1)
#define MAXSTREAM (1<<18)
#define MAXHUFF (2*MAXDIVERSITY)

typedef int bool_t;

#define TRUE 1
#define FALSE 0;

/* huffman tree */
struct huff {
	int freq;
	int value;
	int pleft;
	int pright;
	bool_t isleaf;
};

static int huff_end;
static struct huff hufftree[MAXHUFF];

/* heap */
struct element {
	int freq;
	int huffidx;
};

static int heap_end;
static struct element heap[MAXHEAP];

unsigned char streambuffer[MAXSTREAM];

/* stream */
struct STREAM {
	unsigned long pbit;
	unsigned char *x;
};

typedef struct STREAM stream_t;

/* read only */
struct RO_STREAM {
	unsigned long pbit;
	const unsigned char *x;
};

typedef struct RO_STREAM ro_stream_t;

/* 
|
|	VARIABLES
|
\*---------------------------*/

static int freq[MAXDIVERSITY];
static unsigned code_table[MAXDIVERSITY];
static unsigned size_table[MAXDIVERSITY];
static stream_t Stream = {0, NULL};
static ro_stream_t RO_Stream = {0, NULL};
static const unsigned int VALUEBITS = 8u;

/*==== PROTOTYPES======================================*/


/* heap */

static void freq_init (const unsigned char *in, size_t max);
static void heap_init (void);
static void heap_append (struct element e);
static void heap_sift_up (int x);
static void heap_adjust_down (int top, int last);


/* hufftree */

static int hufftree_from_freq (void);
static int hufftree_from_heap (void);
static void hufftree_to_codes (int start, int n, unsigned code);
static void hufftree_reset (void);
static int hufftree_frombits (ro_stream_t *stream, bool_t *pok);
static void hufftree_tobits (int thisnode, stream_t *stream);
static unsigned int hufftree_readstream (int root, ro_stream_t *s);

/* stream */

/* read only */
static void ro_stream_rewind (ro_stream_t *s);
static void ro_stream_init (ro_stream_t *s, const unsigned char *buffer);
static void ro_stream_done (ro_stream_t *s); 

/* read and write */
static void stream_clear (stream_t *s);
static void stream_init (stream_t *s, unsigned char *buffer);
static void stream_done (stream_t *s); 
static size_t stream_len (stream_t *s);

static void stream_rewind (stream_t *s);
static unsigned int stream_nextbit (ro_stream_t *s);
static unsigned int stream_nextbit_n (ro_stream_t *s, unsigned int width);
static void stream_writebit (stream_t *s, unsigned z);
static void stream_write_n (unsigned code, unsigned width, stream_t *s);		 


static bool_t decode_from_stream (ro_stream_t *stream, size_t n, unsigned char *out);
static void encode_to_stream (const unsigned char *in, size_t inlen, stream_t *stream);

/*static unsigned int stream_next8 (stream_t *s);*/
/*static void stream_write8 (stream_t *s, unsigned z);*/

/* supporting functions */

/*
static void heap_plot (void);
static int fill_block(unsigned char *out);
static char *binstream(unsigned int x, int n);
static void stream_print(stream_t *s, int n);
static void stream_printnext (stream_t *s, int n);
static void stream_dump (stream_t *s, int ori, int n);
static void freq_report (void);
*/

/*=== ENCODE/DECODE=================================================*/

size_t TB_hzip_unused;

static int 
huffman_decode (size_t z, const unsigned char *bz, size_t n, unsigned char *bp)
/* bz:buffer huffman zipped to bp:buffer decoded */
{
	bool_t ok;
	TB_hzip_unused = z; /* to silence compiler */
	ro_stream_init (&RO_Stream, bz);
	ok = decode_from_stream (&RO_Stream, n, bp);
	ro_stream_done (&RO_Stream);
	return ok;
}

static int 
huffman_encode (size_t n, const unsigned char *bp, size_t *z, unsigned char *bz)
/* bz:buffer huffman zipped to bp:buffer decoded */
{
	size_t i, zz;
	stream_init (&Stream, streambuffer);
	encode_to_stream (bp, n, &Stream);
	zz = stream_len (&Stream);
	for (i = 0; i < zz; i++) {
		bz[i] = Stream.x[i];
	}
	*z = zz;
	stream_done (&Stream);
	return TRUE;
}

extern int
huff_decode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	size_t n =     (size_t)in_start[0] 
				| ((size_t)in_start[1] <<  8) 
				| ((size_t)in_start[2] << 16)
				| ((size_t)in_start[3] << 24);	
	TB_hzip_unused = out_max;
	*pout_len = n;
	return huffman_decode (in_len-4, in_start+4, n, out_start);
}


extern int
huff_encode
(const unsigned char *in_start, size_t in_len, unsigned char *out_start, size_t *pout_len, size_t out_max)
{
	bool_t ok;
	size_t hlen = 0;
	TB_hzip_unused = out_max;
	out_start[0] = (unsigned char) ((in_len      ) & 0xffu);
	out_start[1] = (unsigned char) ((in_len >>  8) & 0xffu);
	out_start[2] = (unsigned char) ((in_len >> 16) & 0xffu);
	out_start[3] = (unsigned char) ((in_len >> 24) & 0xffu);
	ok = huffman_encode (in_len, in_start, &hlen, out_start+4);
	*pout_len = hlen + 4;
	return ok;
}


static bool_t
decode_from_stream (ro_stream_t *s, size_t n, unsigned char *out)
{
	int root;
	bool_t ok = TRUE; /* default */

	hufftree_reset ();
	ro_stream_rewind (s);	
	root = hufftree_frombits (s, &ok);
	
	if (ok) {
		while (n-->0) {
			*out++ = (unsigned char) hufftree_readstream (root, s); /* cast to silence compiler */
		}
	}
	return ok;
}


static void
encode_to_stream (const unsigned char *in, size_t inlen, stream_t *stream)
{
	size_t i;
	unsigned x, c, s;
	int root;
	
	stream_clear (&Stream);	
	stream_rewind(&Stream);

	/* pass to collect frequencies */	
	freq_init (in, inlen);

	/* frequency --> heap --> hufftrees */
	root = hufftree_from_freq();	

	/* hufftree --> codes */
	hufftree_to_codes (root, 0, 0);	
	
	/* hufftrees --> stored in bits (stream) */	
	hufftree_tobits (root, stream)	;

	/* input + codes -->  stored in bits (stream) */
	for (i = 0; i < inlen; i++) {
		x = in[i];
		c = code_table[x];
		s = size_table[x];
		stream_write_n (c, s, stream);
	}
	return;
}

/*=== STREAM =================================================*/
/*
static char buffer[256];
*/
/*
static char *
binstream(unsigned int x, int n)
{
	char *s = buffer;
	int i;
	
	for (i = 0; i < n; i++) {
		if (0!=(x&(1<<i))) {
			s[i] = '1';
		} else {
			s[i] = '0';
		}
	}
	s[i] = '\0';
	return buffer;
}
*/

/* READ ONLY */
static void ro_stream_rewind (ro_stream_t *s) {s->pbit = 0; return;}

static void
ro_stream_init (ro_stream_t *s, const unsigned char *buffer)
{ 	
	s->x = buffer;
	s->pbit = 0;
	return;
}

static void
ro_stream_done (ro_stream_t *s)
{ 	
	s->x = NULL;
	s->pbit = 0;
	return;
}


/* READ AND WRITE */
static void stream_rewind (stream_t *s) {s->pbit = 0; return;}

static void
stream_init (stream_t *s, unsigned char *buffer)
{ 	
	s->x = buffer;
	s->pbit = 0;
	return;
}

static void
stream_done (stream_t *s)
{ 	
	s->x = NULL;
	s->pbit = 0;
	return;
}

static void
stream_clear (stream_t *s)
{ 	int i;
	for (i = 0; i < MAXSTREAM; i++) {
		s->x[i] = 0;
	}
	s->pbit = 0;
	return;
}

static size_t
stream_len (stream_t *s)
{
	return	1 + s->pbit/8;
}

static unsigned int
stream_nextbit (ro_stream_t *s)
{
	unsigned long y, byte, bit;	
	y = s->pbit++;
	byte = y / 8;
	bit = y & 7;
	return 1u & (((unsigned)s->x[byte]) >> bit);
}

static unsigned int
stream_nextbit_n (ro_stream_t *s, unsigned int width)
{
	unsigned i;
	unsigned x;
	unsigned r = 0;
	for (i = 0; i < width; i++) {
		x = stream_nextbit (s);		
		r |= (x << i);
	}
	return r;	
}

/*
static unsigned int
stream_next8 (stream_t *s)
{
	unsigned a,b,y,byte,bit;
	y = s->pbit; 
	s->pbit += 8;
	byte = y / 8;
	bit = y & 7;
	a = 0xff & s->x[byte];
	b = 0xff & s->x[byte+1];
	return 0xff & ((a >> bit) | (b << (8-bit)));
}
*/

#if 1
static void
stream_writebit (stream_t *s, unsigned z)
{
	unsigned long y,byte,bit;	
	y = s->pbit++;
	
	byte = y / 8;
	bit = y & 7;
	
	/*	s->x[byte] &= ~(1u << bit);*/
	s->x[byte] = (unsigned char) (s->x[byte] | ((z&1u) << bit));	/* cast to silence compiler */
	return;
}

#else
static void
stream_writebit (stream_t *s, unsigned z)
{
	/* 	This function will write the next bit, 0 or 1 depending on z, and will clear
	|	the following bits (when bit == 0) or some future bytes
	|	Do not use for writing after random access
	|	It is only useful when this function is use for sequential writing on a 
	|	empty buffer.
	*/
	unsigned long y, byte, bit;	
	unsigned char *p;
	y = s->pbit++;

	byte = y / 8;
	bit  = y & 7;
	
	p = &(s->x[byte]);

	/* 	hack to clear the byte only when bit == 0, otherwise, it clears future bytes 
	|	This will avoid clearing the whole buffer beforehand or doing 
	|	*p &= (unsigned char)(~(1u << bit));
	*/
	p[bit] = 0; 

	*p |= (unsigned char)(z&1u) << bit);

	return;
}
#endif


/*
static void
stream_write8 (stream_t *s, unsigned z)
{
	unsigned a,b,c,y,byte,bit;
	y = s->pbit; 
	s->pbit += 8;
	byte = y / 8;
	bit = y & 7;
	a = 0xff & s->x[byte];
	b = 0xff & s->x[byte+1];
	c = a | (b << 8);
	
	c &= ~(0xff << bit);
	c |= z << bit;
	
	s->x[byte] = c & 0xff;	
	s->x[byte+1] = 0xff & (c >> 8);
	
	return;
}
*/

static void
stream_write_n (unsigned code, unsigned width, stream_t *s)		 
{
	unsigned i;
	for (i = 0; i < width; i++) {
		stream_writebit (s, 1u & (code >> i));
	}
	return;
}

/*
static void
stream_printnext (stream_t *s, int n)
{
	int i; unsigned int x;  
	unsigned long int oldpos = s->pbit;
	
	printf("\n");	
	for (i = 0; i < n; i++)  {
			if ((i & 7) == 0)
				printf ("\n");		
			x = stream_next8 (s);
			printf ("%s ", binstream(x,8) ); 
	}
	printf("\n");

	s->pbit = oldpos;
	return;
}
*/

/*
static void
stream_dump (stream_t *s, int ori, int n)
{
	int i; unsigned int x;  
	unsigned long int oldpos = s->pbit;

	s->pbit = ori;
	
	printf("\n");	
	for (i = 0; i < n; i++)  {
			if ((i & 7) == 0)
				printf ("\n");		
			x = stream_next8 (s);
			printf ("%s ", binstream(x,8) ); 
	}
	printf("\n");

	s->pbit = oldpos;
	return;
}
*/

/*=== HUFFTREE=================================================*/

#define LEFTCODE 0u
#define RIGHTCODE 1u
#define BITLEAF 1
#define BITNODE 0

static void
hufftree_reset (void)
{
	struct huff h;
	int i;
	for (i = 0; i < 2*MAXDIVERSITY; i++) {
		h.isleaf = FALSE;
		h.value = 0;
		h.freq = 0;
		h.pleft = 0;
		h.pright = 0;
		hufftree[i] = h;
	}
	huff_end = 0;
}

static int
hufftree_from_heap (void)
{
	int top, newidx /*, left, right, lesser */ ;
	struct huff h;
	
	for (;;)
	{	
		if (heap_end == 2) { /* at least top element */
			/* done */
			break;
		}
	
		/* work at the top */
		top = 1;	
/*	
		left = 2*top;  
		right = left + 1;

		lesser = left;
		if (right < heap_end && (heap[right].freq < heap[left].freq))
			lesser = right;
*/
		/* new huff node */
		newidx = huff_end++;
	
		h.isleaf = FALSE;
		h.value = -1;
		h.freq =  heap[top].freq; /* will be incremented later when in 'combine' */
		h.pleft = heap[top].huffidx;
		h.pright = -1; /* will be attached the next element */
	

	#ifdef TRACE
	printf ("\n\nBefore Eliminate Top\n");	
	heap_plot();	
	#endif
	
		/* eliminate top */
		heap[top] = heap[--heap_end];

		/* next 'lesser' element at 'top' */
		heap_adjust_down (1, heap_end-1);	

	#ifdef TRACE
	printf ("\n\nEliminate Top\n");	
	heap_plot ();	
	#endif
	
		/* combine */
		h.pright = heap[1].huffidx;
		h.freq += heap[1].freq; /* combine frequencies */
		hufftree[newidx] = h;

		heap[1].freq = h.freq; 
		heap[1].huffidx = newidx;
		
		/* adjust the combined elements */
		heap_adjust_down (1, heap_end-1);	

	#ifdef TRACE	
	printf ("\n\nAfter Combine\n");
	heap_plot ();	
	#endif
	
	}

	return heap[1].huffidx;
}


static void
hufftree_to_codes (int start, int n, unsigned code)
{
	int x, m;
	unsigned c;
	int value;	
	
	#ifdef TRACK	
	if (n == 0)
		printf ("\nHufftree to codes\n"); 
	#endif

	assert (n >= 0);	
	
	x = hufftree[start].pleft;
	c = code | (LEFTCODE << n);
	m = n + 1;
	
	/* LEFT */
	if (hufftree[x].isleaf) {
		value = hufftree[x].value;	
		code_table[value] = c;
		size_table[value] = (unsigned)m; 

		#ifdef TRACK	
		printf ("value=%c:%d, code=%d \"%s\", size=%d\n", value,value, c, binstream(c,m), m); 
		#endif
		
	} else {
		hufftree_to_codes(x, m, c);
	}	

	
	/* RIGHT */	
	x = hufftree[start].pright;
	c = code | (RIGHTCODE << n);
	m = n + 1;

	if (hufftree[x].isleaf) {
		value = hufftree[x].value;			
		code_table[value] = c;
		size_table[value] = (unsigned)m;

		#ifdef TRACK	
		printf ("value=%c:%d, code=%d \"%s\", size=%d\n", value,value, c, binstream(c,m), m);
		#endif	
		
	} else {
		hufftree_to_codes(x, m, c);
	}		

	return;
}


static int
hufftree_frombits (ro_stream_t *stream, bool_t *pok)
{
	unsigned bit;
	unsigned value;
	int thisnode;
	struct huff h;
	
	if (!*pok)
		return 0;

	bit = stream_nextbit(stream);
	if (bit == BITLEAF) {
		/* leaf */
		value = stream_nextbit_n (stream, VALUEBITS);
		thisnode = huff_end++;
		h.isleaf = TRUE;
		h.value = (int)value;
		h.freq =  0;
		h.pleft = 0;
		h.pright = 0;

		if (thisnode >= MAXHUFF) {
			*pok = FALSE;
			return 0;	
		}

		hufftree[thisnode] = h;

		#ifdef TRACK	
		printf ("Huff leaf, %d=%c\n", value, value); 
		#endif
		
		return thisnode;		
		
	} else {
		/* node */
		thisnode = huff_end++;

		if (thisnode >= MAXHUFF) {
			*pok = FALSE;
			return 0;
		}

		h.isleaf = FALSE;
		h.value = -1;
		h.freq =  0;
		h.pleft = hufftree_frombits (stream, pok);	
		h.pright = hufftree_frombits (stream, pok);
		hufftree[thisnode] = h;		
		return thisnode;
	}
}


static void
hufftree_tobits (int thisnode, stream_t *stream)
{

	if (hufftree[thisnode].isleaf) {

		#ifdef TRACK			
		{int c = hufftree[thisnode].value; printf ("[leaf=1][%c:%d=%s]", c, c, binstream(c,8));} 
		#endif
		
		assert (0 <= hufftree[thisnode].value);

		stream_writebit (stream, BITLEAF);
		stream_write_n ((unsigned)hufftree[thisnode].value, VALUEBITS, stream);
		
	} else {
		stream_writebit (stream, BITNODE);		

		#ifdef TRACK			
		printf ("[node=0]");	
		#endif
		
		hufftree_tobits (hufftree[thisnode].pleft, stream);
		hufftree_tobits (hufftree[thisnode].pright, stream);		
		
	}
	return;
}


static unsigned int
hufftree_readstream (int root, ro_stream_t *s)
{
	unsigned bit;
	int next;
	
	bit = stream_nextbit(s);
	if (bit == RIGHTCODE) {
		/* right */
		next = hufftree[root].pright;
	} else {
		/*ASSERT (bit == LEFTCODE */
		/* left */
		next = hufftree[root].pleft;
	}

	if (hufftree[next].isleaf) {
		assert (0 <= hufftree[next].value);
		return (unsigned)hufftree[next].value;
	} else {
		return hufftree_readstream (next, s);
	}
}

/*==== HEAP ==========================================*/

static void
heap_init (void)
{
	heap_end = 1;
	return;
}

static void
heap_append (struct element e)
{
	/*ASSERT (heap_end < MAXHEAP);*/
	heap[heap_end++] = e;
	return;
}

static void
heap_sift_up (int x)
{
	struct element t;
	int p;
	int c = x;
	while (c > 1) {
		p = c / 2;
		if (heap[c].freq < heap[p].freq) {
			t = heap[c]; heap[c] = heap[p]; heap[p] = t;
		} else {
			break;
		}
		c = p;
	}
	return;
}



static void
heap_adjust_down (int top, int last)
{
	struct element t;
	int p;
	int c;
	int left, right;
	
	if (last == top) { /* at least top element */
		/* done */
		return;
	}	
	
	/* starts at the top */	
	p = top;
	
	while (p <= last)
	{	
		left = 2*p;  
		right = left + 1;
	
		if (left > last)
			break;
		
		if (right <= last && (heap[right].freq < heap[left].freq))
			c = right;
		else
			c = left;		

		if (c > last)
			break;
		
		if (heap[c].freq < heap[p].freq) {
			t = heap[c]; heap[c] = heap[p]; heap[p] = t;
		} else {
			break;
		}
		p = c;
	}	
	return;
}


/*
static void
heap_plot (void)
{
	unsigned int line, limit, j;
	int n = heap_end;
	printf("===========================\n");	
	line = 1;
	j = 1;
	while (j < n) {
		limit = 1 << line;
		while (j < limit && j < n) {
			printf("%3d:%c ",heap[j].freq, hufftree[heap[j].huffidx].value);
			j++;
		}
		while (j < limit) {
			printf("%3s ","--");
			j++;
		}		
		line++; printf("\n");
	}
	printf("===========================\n");	
	return;
}
*/

/*
static void
freq_report (void)
{
	int i;
	printf ("\nFREQUENCIES\n");
	for (i = 0; i < MAXDIVERSITY; i++) {
		if (freq[i] > 0) {
			printf ("%c: %2d: %d: %d\n", i, i, freq[i], code_table[i]);
		}
	}	
	printf ("\n");
	return;
}
*/

static void
freq_init (const unsigned char *in, size_t max)
{
	size_t i;
	
	/* clean up frequencies */
	for (i = 0; i < MAXDIVERSITY; i++) {
		freq [i] = 0;
		code_table[i] = 0;
		size_table[i] = 0;
	}

	/* build frequencies */
	for (i = 0; i < max; i++) {
		freq [in[i]]++;
	}

	#ifdef TRACK
	freq_report();
	#endif
	
	return;
}

static int
hufftree_from_freq (void)
{
	int i;
	struct huff h;
	struct element e;
	int root;
	
	hufftree_reset ();	
	
	/* build huff tree elements */
	huff_end = 0;
	for (i = 0; i < MAXDIVERSITY; i++) {
		if (freq[i] > 0) {
			h.isleaf = TRUE;
			h.value = i;
			h.freq =  freq[i];
			h.pleft = 0;
			h.pright = 0;
			hufftree[huff_end++] = h;
		}
	}

	/* build heap */
	heap_init();
	for (i = 0; i < huff_end; i++) {
		e.freq =  hufftree[i].freq;
		e.huffidx = i;
		heap_append(e);
		heap_sift_up(heap_end-1);
	}
	
	#ifdef TRACE
	heap_plot ();
	#endif
	
	root = hufftree_from_heap();
	
	/*hufftree_to_codes (root, 0, 0);*/

	return root;
}



