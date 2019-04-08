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

#include <stdlib.h>
#include <stdio.h>
#include "gtb-att.h"

#if 0

#include "mtypes.h" 
#include "bool_t.h" 
#include "maindef.h" 

#else

/* mtypes.h */

typedef unsigned int SQUARE;
typedef unsigned char SQ_CONTENT;

/* bool_t.h */

#if !defined(bool_t)
typedef int						bool_t;
#endif

#if !defined(TRUE)
#define TRUE ((bool_t)1)
#endif

#if !defined(FALSE)
#define FALSE ((bool_t)0)
#endif

/* maindef.h */

#define NOPIECE 0u
#define PAWN    1u
#define KNIGHT  2u
#define BISHOP  3u
#define ROOK    4u
#define QUEEN   5u
#define KING    6u
#define PIECE_MASK (KING|PAWN|KNIGHT|BISHOP|ROOK|QUEEN)

/*Whites*/
#define wK (KING   | WHITES)
#define wP (PAWN   | WHITES)
#define wN (KNIGHT | WHITES)
#define wB (BISHOP | WHITES)
#define wR (ROOK   | WHITES)
#define wQ (QUEEN  | WHITES)

/*Blacks*/
#define bK (KING   | BLACKS)
#define bP (PAWN   | BLACKS)
#define bN (KNIGHT | BLACKS)
#define bB (BISHOP | BLACKS)
#define bR (ROOK   | BLACKS)
#define bQ (QUEEN  | BLACKS)

/*Bits that define color */

#define WHITES (1u<<6)
#define BLACKS (1u<<7)

/*squares*/
enum SQUARES {
	A1,B1,C1,D1,E1,F1,G1,H1,
	A2,B2,C2,D2,E2,F2,G2,H2,
	A3,B3,C3,D3,E3,F3,G3,H3,
	A4,B4,C4,D4,E4,F4,G4,H4,
	A5,B5,C5,D5,E5,F5,G5,H5,
	A6,B6,C6,D6,E6,F6,G6,H6,
	A7,B7,C7,D7,E7,F7,G7,H7,
	A8,B8,C8,D8,E8,F8,G8,H8,
	NOSQUARE,
	ERRSQUARE = 128
};
#endif

/*----------------------------------------------------------------------*/

#ifndef NDEBUG
#define NDEBUG
#endif
#ifdef DEBUG
#undef NDEBUG
#endif
#include "assert.h"

/*----------------------------------------------------------------------*/

/* global variables */
uint64_t Reach [7] [64];

/* static variables */
static unsigned char	attmap [64] [64];
static unsigned int		attmsk [256];

/* static functions */
static unsigned int mapx88 (unsigned int x);

/* macros */
#define BB_ISBITON(bb,bit)   (0 != (((bb)>>(bit)) & U64(1)))

#define map88(x)    (   (x) + ((x)&070)        )
#define unmap88(x)  ( ( (x) + ((x)& 07) ) >> 1 )

/*----------------------------------------------------------------------*/

static unsigned int
mapx88 (unsigned int x)
{
	return ((x & 070) << 1) | (x & 07);
}


void
attack_maps_init(void)
{
	int i;
	unsigned int m, from, to;
	unsigned int to88, fr88;
	int diff;	

	uint64_t rook, bishop, queen, knight, king;

	if (!reach_was_initialized()) {
		printf ("Wrong initialization order of data\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < 256; ++i) {
		attmsk [i] = 0;
	}
	attmsk[wP] 		= 1 << 0;
	attmsk[bP] 		= 1 << 1;

	attmsk[KNIGHT] 	= 1 << 2;
	attmsk[wN] 		= 1 << 2;
	attmsk[bN] 		= 1 << 2;

	attmsk[BISHOP] 	= 1 << 3;
	attmsk[wB] 		= 1 << 3;
	attmsk[bB] 		= 1 << 3;

	attmsk[ROOK  ] 	= 1 << 4;
	attmsk[wR] 		= 1 << 4;
	attmsk[bR] 		= 1 << 4;

	attmsk[QUEEN ] 	= 1 << 5;
	attmsk[wQ] 		= 1 << 5;
	attmsk[bQ] 		= 1 << 5;

	attmsk[KING  ] 	= 1 << 6;
	attmsk[wK] 		= 1 << 6;
	attmsk[bK] 		= 1 << 6;

	for (to = 0; to < 64; ++to) {
		for (from = 0; from < 64; ++from) {
			m = 0;
			rook   = Reach [ROOK]   [from]; 
			bishop = Reach [BISHOP] [from]; 
			queen  = Reach [QUEEN]  [from]; 
			knight = Reach [KNIGHT] [from];
			king   = Reach [KING]   [from]; 

			if (BB_ISBITON (knight, to)) {
				m |= attmsk[wN];
			}
			if (BB_ISBITON (king, to)) {
				m |= attmsk[wK];
			}
			if (BB_ISBITON (rook, to)) {
				m |= attmsk[wR];
			}
			if (BB_ISBITON (bishop, to)) {
				m |= attmsk[wB];
			}			
			if (BB_ISBITON (queen, to)) {
				m |= attmsk[wQ];
			}
			
			to88 = mapx88(to);
			fr88 = mapx88(from);
			diff = (int)to88 - (int)fr88;

			if (diff ==  17 || diff ==  15) {
				m |= attmsk[wP];
			}
			if (diff == -17 || diff == -15) {
				m |= attmsk[bP];
			}

			attmap [to] [from] = (unsigned char) m;
		}		
	}

}

bool_t
possible_attack(unsigned int from, unsigned int to, unsigned int piece)
{

	assert (piece < 256);
	assert (from < 64 && to < 64);
	assert (reach_was_initialized());
	assert (attmsk [piece] != 0 || 0==fprintf(stderr, "PIECE=%d\n",piece) ); /* means piece has been considered */ 

	return 0 != (attmap [to] [from] & attmsk [piece]);
}

/*
|
|	REACH ROUTINES
|
\*----------------------------------------------*/


enum Key {REACH_INITIALIZED_KEY = 0x1313};
static int reach_initialized_key = 0;

bool_t
reach_was_initialized (void)
{
	return	reach_initialized_key == REACH_INITIALIZED_KEY;
}

void
reach_init (void)
{	
	SQUARE buflist[64+1], *list;
	SQ_CONTENT pc;
	int stp_a [] = {15, -15 };
	int stp_b [] = {17, -17 };
	int STEP_A, STEP_B;
	unsigned int side;
	unsigned int index;
	SQUARE sq, us;

	int s;

	for (pc = KNIGHT; pc < (KING+1); pc++) {
		for (sq = 0; sq < 64; sq++) {
			uint64_t bb = U64(0x0);
			tolist_rev (U64(0x0), pc, sq, buflist);
			for (list = buflist; *list != NOSQUARE; list++) {
				bb |= U64(1) << *list;
			}
			Reach [pc] [sq] = bb;
		}
	}

	for (side = 0; side < 2; side++) {
		index  = 1u ^ side;
		STEP_A = stp_a[side];
		STEP_B = stp_b[side];
		for (sq = 0; sq < 64; sq++) {

			int sq88 = (int)map88(sq);
			uint64_t bb = U64(0x0);

			list = buflist;
	

			s = sq88 + STEP_A;
			if (0 == (s & 0x88)) {
				us = (SQUARE)unmap88(s);
				*list++ = us;
			}
			s = sq88 + STEP_B;
			if (0 == (s & 0x88)) {
				us = (SQUARE)unmap88(s);
				*list++ = us;
			}
			*list = NOSQUARE;

			for (list = buflist; *list != NOSQUARE; list++) {
				bb |= U64(1) << *list;
			}
			Reach [index] [sq] = bb;
		}
	}
	reach_initialized_key = REACH_INITIALIZED_KEY;
}

/*--------------------------------------------------------------------------------*/

static const int bstep[]  = { 17,  15, -15, -17,  0};
static const int rstep[]  = {  1,  16,  -1, -16,  0};
static const int nstep[]  = { 18,  33,  31,  14, -18, -33, -31, -14,  0};
static const int kstep[]  = {  1,  17,  16,  15,  -1, -17, -16, -15,  0};
	 
static const 
int *psteparr[] = {NULL, NULL, /* NOPIECE & PAWN */
                   nstep, bstep, rstep, kstep, kstep /* same for Q & K*/
                  };
static const 
int   pslider[] = {FALSE, FALSE,
                   FALSE,  TRUE,  TRUE,  TRUE, FALSE
                  };
	
void 
tolist_rev (uint64_t occ, SQ_CONTENT input_piece, SQUARE sq, SQUARE *list)
/* reversible moves from pieces. Output is a list of squares */
{
   	int direction;
   	unsigned int pc;
    int s;
   	int from;
	int step;
	const int *steparr;
	bool_t slider;
	SQUARE us;

    assert (sq < 64);

	/* i.e. no pawn allowed as input */
	assert (input_piece == KNIGHT || input_piece == BISHOP ||
			input_piece == ROOK   || input_piece == QUEEN  || 
			input_piece == KING);

   	from = (int)map88(sq);
   	
	pc = input_piece & (PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING);

	steparr = psteparr [pc];
	slider  = pslider  [pc];

	if (slider) {
	 	
		for (direction = 0; steparr[direction] != 0; direction++) {
			step = steparr[direction];
			s = from + step;
			while (0 == (s & 0x88)) {
				us = (SQUARE)unmap88(s);
				if (0 != (0x1u & (unsigned int)(occ >> us)))
					break;
				*list++ = us;
				s += step;
  			}
 		}

	} else {
		
		for (direction = 0; steparr[direction] != 0; direction++) {
			step = steparr[direction];
			s = from + step;
			if (0 == (s & 0x88)) {
				us = (SQUARE)unmap88(s);
				if (0 == (0x1u & (unsigned int)(occ >> us))) {
					*list++ = us;
				}
  			}
 		}		
	}

	*list = NOSQUARE;

   	return;
}	



