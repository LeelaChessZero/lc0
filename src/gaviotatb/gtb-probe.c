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


/* NBBOTF will remove the internal bitbase on the fly */
#ifdef NBBOTF
	#ifdef WDL_PROBE
		#undef WDL_PROBE
	#endif
#else
	#define WDL_PROBE
#endif

/*-- Intended to be modified to make public --> Supporting functions the TB generator ---------------------*/

#ifdef GTB_SHARE
#define SHARED_forbuilding
#endif

/*---------------------------------------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gtb-probe.h"

#if defined(SHARED_forbuilding)
	#include "gtb-prob2.h"
#else
	#define mySHARED static
	typedef unsigned char			SQ_CONTENT;
	typedef unsigned int 			SQUARE; 
#endif

#include "sysport.h"
#include "gtb-att.h"
#include "gtb-types.h"

/*---------------------------------------------------------------------------------------------------------*/
/*#include "posit_t.h"*/

#define MAX_LISTSIZE 17
#if 0
typedef unsigned 		sq_t;
typedef unsigned char 	pc_t;
typedef uint32_t		mv_t;
#endif

struct posit {
	sq_t 			ws[MAX_LISTSIZE];
	sq_t 			bs[MAX_LISTSIZE];
	pc_t 			wp[MAX_LISTSIZE];
	pc_t 			bp[MAX_LISTSIZE];
	sq_t 			ep;
	unsigned int 	stm;
	unsigned int 	cas;
};
typedef struct 	posit posit_t;

#if 0
typedef long int		tbkey_t;
#endif

/*---------------------------------------------------------------------------------------------------------*/
/*#include "bool_t.h"*/

#if !defined(H_BOOL)
typedef int						bool_t;
#endif

#if !defined(TRUE)
#define TRUE ((bool_t)1)
#endif

#if !defined(FALSE)
#define FALSE ((bool_t)0)
#endif

/*--------- private if external building code is not present ----------------------------------------------*/

#if !defined(SHARED_forbuilding)

#define MAX_EGKEYS 145
#define SLOTSIZE 1
#define NOINDEX ((index_t)(-1))

#if 0
typedef unsigned short int 	dtm_t;
typedef size_t 				index_t;
/*typedef int 				index_t;*/
#endif

enum Loading_status {	
				STATUS_ABSENT 		= 0, 
				STATUS_STATICRAM 	= 1, 
				STATUS_MALLOC 		= 2,
				STATUS_FILE   		= 3, 
				STATUS_REJECT 		= 4
};

struct endgamekey {
	int 		id;
	const char *str;
	index_t 	maxindex;
	index_t 	slice_n;
	void   		(*itopc) (index_t, SQUARE *, SQUARE *);
	bool_t 		(*pctoi) (const SQUARE *, const SQUARE *, index_t *);
	dtm_t *		egt_w;
	dtm_t *		egt_b;
	FILE *		fd;
	int 		status;
	int			pathn; 
};
#endif

/*----------------------------------------------------------------------------------------------------------*/

/* array for better moves */
#ifdef GTB_SHARE
mySHARED int		bettarr [2] [8] [8];
#endif

/*------------ ENUMS ----------------------------------------------------------*/

enum Mask_values {	
					RESMASK  = tb_RESMASK, 
					INFOMASK = tb_INFOMASK, 
					PLYSHIFT = tb_PLYSHIFT  
};

enum Info_values {	
					iDRAW    = tb_DRAW, 
					iWMATE   = tb_WMATE, 
					iBMATE   = tb_BMATE, 
					iFORBID  = tb_FORBID,
		 
					iDRAWt   = tb_DRAW  |4, 
					iWMATEt  = tb_WMATE |4, 
					iBMATEt  = tb_BMATE |4, 
					iUNKNOWN = tb_UNKNOWN,

					iUNKNBIT = (1<<2)
};

/*-------------------------- inherited from a previous maindef.h -----------*/

#define WHITES (1u<<6)
#define BLACKS (1u<<7)

#define NOPIECE 0u
#define PAWN    1u
#define KNIGHT  2u
#define BISHOP  3u
#define ROOK    4u
#define QUEEN   5u
#define KING    6u

#define WH 0
#define BL 1
#define Opp(x) ((x)^1)
#define wK (KING   | WHITES)

/*-------------------
       SQUARES
  -------------------*/

/* from 1-63 different squares posibles   */

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

/*------------------- end of inherited from a previous maindef.h -----------*/
 
#if !defined(NDEBUG)
#define NDEBUG
#endif
#ifdef DEBUG
#undef NDEBUG
#endif
#include "assert.h"

/*------------------- general DEFINES--------------------------- -----------*/

#define gtbNOSIDE	((unsigned)-1)
#define gtbNOINDEX	((index_t)-1)

/*************************************************\
|
|				COMPRESSION SCHEMES 
|
\*************************************************/

#include "gtb-dec.h"

static const char *const Extension[] = {
							 ".gtb.cp0"
							,".gtb.cp1"
							,".gtb.cp2"
							,".gtb.cp3"
							,".gtb.cp4"
							,".gtb.cp5"
							,".gtb.cp6"
							,".gtb.cp7"
							,".gtb.cp8"
							,".gtb.cp9"
};

/*************************************************\
|
|					MOVES 
|
\*************************************************/

enum move_kind {
		NORMAL_MOVE = 0,
		CASTLE_MOVE,
		PASSNT_MOVE,
		PROMOT_MOVE
};

enum move_content {
		NOMOVE = 0	
};

#define MV_TYPE(mv)   ( (BYTE)       ((mv) >>6 & 3 )      )
#define MV_TO(mv)     ( (SQUARE)     ((mv) >>8 & 63)      )
#define MV_PT(mv)     ( (SQ_CONTENT) ((mv) >>(3+16) &7  ) )
#define MV_TK(mv)     ( (SQ_CONTENT) ((mv) >>(6+16) &7  ) )
#define MV_FROM(mv)   ( (SQUARE)     ((mv)     & 63)      )

/*
|   move,type,color,piece,from,to,taken,promoted
*------------------------------------------------------------------*/

#define MV_BUILD(mv,ty,co,pc,fr,to,tk,pm) (                        \
    (mv)    =  (fr)     | (to)<< 8      | (ty)<<  6     | (co)<<8  \
            |  (pc)<<16 | (pm)<< (3+16) | (tk)<< (6+16)            \
)

#define MV_ADD_TOTK(mv,to,tk) (          \
     mv     |= (uint32_t)(to) << 8       \
            |  (uint32_t)(tk) << (6+16)  \
)

#define map88(x)    (   (x) + ((x)&070)        )
#define unmap88(x)  ( ( (x) + ((x)& 07) ) >> 1 )

/*************************************************\
|
|				STATIC VARIABLES 
|
\*************************************************/

static int GTB_scheme = 4;

/*************************************************\
|
|	needed for 
|	PRE LOAD CACHE AND DEPENDENT FUNCTIONS 
|
\*************************************************/

#define EGTB_MAXBLOCKSIZE 65536

static int GTB_MAXOPEN = 4;

static bool_t 			Uncompressed = TRUE;
static unsigned char 	Buffer_zipped [EGTB_MAXBLOCKSIZE];
static unsigned char 	Buffer_packed [EGTB_MAXBLOCKSIZE];
static unsigned int		zipinfo_init (void);
static void 			zipinfo_done (void);

enum Flip_flags {
		WE_FLAG = 1, NS_FLAG = 2,  NW_SE_FLAG = 4
}; /* used in flipt */

struct filesopen {
		int n;
		tbkey_t *key;
};

/* STATIC GLOBALS */

static struct filesopen	fd = {0, NULL};

static bool_t 			TB_INITIALIZED = FALSE;
static bool_t			DTM_CACHE_INITIALIZED = FALSE;

static int				WDL_FRACTION = 64;
static int				WDL_FRACTION_MAX = 128;
	
static size_t			DTM_cache_size = 0;
static size_t			WDL_cache_size = 0;

static unsigned int		TB_AVAILABILITY = 0;

/* LOCKS */
static mythread_mutex_t	Egtb_lock;


/****************************************************************************\
 *
 *
 *			DEBUGGING or PRINTING ZONE
 *
 *
 ****************************************************************************/

#if 0
#define FOLLOW_EGTB
#ifndef DEBUG
#define DEBUG
#endif
#endif

#define validsq(x) ((x) >= A1 && (x) <= H8)

#if defined(DEBUG)
static void 	print_pos (const sq_t *ws, const sq_t *bs, const pc_t *wp, const pc_t *bp);
#endif

#if defined(DEBUG) || defined(FOLLOW_EGTB)
static void 	output_state (unsigned stm, const SQUARE *wSQ, const SQUARE *bSQ, 
								const SQ_CONTENT *wPC, const SQ_CONTENT *bPC);
static const char *Square_str[64] = {
 	"a1","b1","c1","d1","e1","f1","g1","h1",
 	"a2","b2","c2","d2","e2","f2","g2","h2",
 	"a3","b3","c3","d3","e3","f3","g3","h3",
 	"a4","b4","c4","d4","e4","f4","g4","h4",
 	"a5","b5","c5","d5","e5","f5","g5","h5",
 	"a6","b6","c6","d6","e6","f6","g6","h6",
 	"a7","b7","c7","d7","e7","f7","g7","h7",
 	"a8","b8","c8","d8","e8","f8","g8","h8"
};
static const char *P_str[] = {
	"--", "P", "N", "B", "R", "Q", "K"
};
#endif

#ifdef FOLLOW_EGTB
	#define STAB
	#define STABCONDITION 1 /*(stm == BL && whiteSQ[0]==H1 && whiteSQ[1]==D1 && whiteSQ[2]==D3 && blackSQ[0]==C2 )*/
	static bool_t GLOB_REPORT = TRUE;
#endif

#if defined(FOLLOW_EGTB)
static const char *Info_str[8] = {	
	" Draw", " Wmate", " Bmate", "Illegal", 
	"~Draw", "~Wmate", "~Bmate", "Unknown" 
};
#endif

static void		list_index (void);
static void 	fatal_error(void) {
    exit(EXIT_FAILURE);
}

#ifdef STAB
	#define FOLLOW_LU(x,y)  {if (GLOB_REPORT) printf ("************** %s: %lu\n", (x), (long unsigned)(y));}
#else
	#define FOLLOW_LU(x,y)
#endif

#ifdef STAB
	#define FOLLOW_LULU(x,y,z)  {if (GLOB_REPORT) printf ("************** %s: %lu, %lu\n", (x), (long unsigned)(y), (long unsigned)(z));}
#else
	#define FOLLOW_LULU(x,y,z)
#endif

#ifdef STAB
	#define FOLLOW_label(x) {if (GLOB_REPORT) printf ("************** %s\n", (x));}
#else
	#define FOLLOW_label(x)
#endif

#ifdef STAB
	#define FOLLOW_DTM(msg,dtm)  {if (GLOB_REPORT) printf ("************** %s: %lu, info:%s, plies:%lu \n"\
	, (msg), (long unsigned)(dtm), (Info_str[(dtm)&INFOMASK]), (long unsigned)((dtm)>>PLYSHIFT)\
	);}
#else
	#define FOLLOW_DTM(msg,dtm)
#endif


/*--------------------------------*\
|
|
|		INDEXING FUNCTIONS
|
|
*---------------------------------*/

#define IDX_set_empty(x) {x=0;x--;}
#define IDX_is_empty(x) (0==(1+(x)))

#define NO_KKINDEX NOINDEX
#define MAX_KKINDEX 462
#define MAX_PPINDEX 576
#define MAX_PpINDEX (24 * 48)
/*1128*/
#define MAX_AAINDEX ((63-62) + (62 * (127-62)/2) - 1 + 1)
#define MAX_AAAINDEX (64*21*31)
#define MAX_PP48_INDEX (1128)
/* (24*23*22/6) + 24 * (24*23/2) */
#define MAX_PPP48_INDEX 8648

/* VARIABLES */

static index_t			kkidx [64] [64];
static index_t			ppidx [24] [48];
static index_t 			pp48_idx[48][48];
static index_t 			ppp48_idx[48][48][48];

static sq_t				wksq [MAX_KKINDEX];
static sq_t				bksq [MAX_KKINDEX];
static sq_t				pp48_sq_x[MAX_PP48_INDEX];
static sq_t				pp48_sq_y[MAX_PP48_INDEX]; 

static index_t		 	pp_hi24 [MAX_PPINDEX]; /* was unsigned int */
static index_t		 	pp_lo48 [MAX_PPINDEX];
static unsigned int 	flipt [64] [64];
static index_t		 	aaidx [64] [64]; /* was unsigned int */
static unsigned char 	aabase [MAX_AAINDEX];

static uint8_t			ppp48_sq_x[MAX_PPP48_INDEX];
static uint8_t			ppp48_sq_y[MAX_PPP48_INDEX]; 
static uint8_t			ppp48_sq_z[MAX_PPP48_INDEX]; 

/* FUNCTIONS */

static void 	init_indexing (int verbosity);
static void 	norm_kkindex (SQUARE x, SQUARE y, /*@out@*/ SQUARE *pi, /*@out@*/ SQUARE *pj);
static void 	pp_putanchorfirst (SQUARE a, SQUARE b, /*@out@*/ SQUARE *out_anchor, /*@out@*/ SQUARE *out_loosen);

static index_t	wsq_to_pidx24 (SQUARE pawn);
static index_t 	wsq_to_pidx48 (SQUARE pawn);
static SQUARE 	pidx24_to_wsq (index_t a);
static SQUARE 	pidx48_to_wsq (index_t a);

static SQUARE 	flipWE    		(SQUARE x) { return x ^  07;}
static SQUARE 	flipNS    		(SQUARE x) { return x ^ 070;}
static SQUARE 	flipNW_SE 		(SQUARE x) { return ((x&7)<<3) | (x>>3);}
static SQUARE 	getcol    		(SQUARE x) { return x &  7;}
static SQUARE 	getrow    		(SQUARE x) { return x >> 3;}
static bool_t 	in_queenside	(sq_t x)   { return 0 == (x & (1<<2));}

/* 1:0 */
static void 	kxk_indextopc   (index_t i, SQUARE *pw, SQUARE *pb);

/* 2:0 */
static void 	kabk_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);
static void 	kakb_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);
static void		kaak_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);

/* 2:1 */
static void 	kabkc_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kaakb_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

/* 3:0 */
static void 	kabck_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kaabk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kaaak_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kabbk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

/* one pawn */
static void 	kpk_indextopc   (index_t i, SQUARE *pw, SQUARE *pb);

/* 1:1 one pawn */
static void 	kakp_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);

/* 2:0 one pawn */
static void 	kapk_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);

/* 2:0 two pawns */
static void 	kppk_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);

/*  2:1 one pawn */
static void		kapkb_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kabkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kaakp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

/*  2:1 + 3:0 two pawns */
static void		kppka_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kappk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kapkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

/*  3:0 one pawn */
static void		kabpk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kaapk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

/*  three pawns */
static void		kpppk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);
static void		kppkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

/* 1:1 two pawns */
static void 	kpkp_indextopc  (index_t i, SQUARE *pw, SQUARE *pb);

/* corresponding pc to index */
static bool_t 	kxk_pctoindex   (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kabk_pctoindex  (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kakb_pctoindex  (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kpk_pctoindex   (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kakp_pctoindex  (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kapk_pctoindex  (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t   kppk_pctoindex  (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kaak_pctoindex  (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kabkc_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);

static bool_t 	kaakb_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);/**/

static bool_t 	kabck_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kaabk_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);/**/
static bool_t 	kaaak_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kabbk_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);/**/
static bool_t 	kapkb_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kabkp_pctoindex (const SQUARE *pw, const SQUARE *pb, /*@out@*/ index_t *out);
static bool_t 	kaakp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kppka_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out);
static bool_t 	kappk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kapkp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kabpk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kaapk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kppkp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kpppk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out);
static bool_t 	kpkp_pctoindex  (const SQUARE *pw,     const SQUARE *pb, /*@out@*/ index_t *out);

/* testing functions */
static bool_t 	test_kppk  (void);
static bool_t 	test_kaakb (void);
static bool_t 	test_kaabk (void);
static bool_t 	test_kaaak (void);
static bool_t 	test_kabbk (void);
static bool_t 	test_kapkb (void);
static bool_t 	test_kabkp (void);
static bool_t 	test_kppka (void);
static bool_t 	test_kappk (void);
static bool_t 	test_kapkp (void);
static bool_t 	test_kabpk (void);
static bool_t 	test_kaapk (void);
static bool_t 	test_kaakp (void);
static bool_t 	test_kppkp (void);
static bool_t 	test_kpppk (void);

static unsigned flip_type (SQUARE x, SQUARE y);
static index_t 	init_kkidx (void);
static index_t	init_ppidx (void);
static void    	init_flipt (void);
static index_t 	init_aaidx (void);
static index_t	init_aaa   (void);
static index_t	init_pp48_idx (void);
static index_t	init_ppp48_idx (void);

enum TB_INDEXES
	 {	 MAX_KXK 	= MAX_KKINDEX*64 
		,MAX_kabk 	= MAX_KKINDEX*64*64 
		,MAX_kakb 	= MAX_KKINDEX*64*64
		,MAX_kpk	= 24*64*64
		,MAX_kakp	= 24*64*64*64
		,MAX_kapk	= 24*64*64*64		
		,MAX_kppk	= MAX_PPINDEX*64*64
		,MAX_kpkp	= MAX_PpINDEX*64*64
		,MAX_kaak	= MAX_KKINDEX*MAX_AAINDEX
		,MAX_kabkc	= MAX_KKINDEX*64*64*64
		,MAX_kabck	= MAX_KKINDEX*64*64*64
		,MAX_kaakb	= MAX_KKINDEX*MAX_AAINDEX*64
		,MAX_kaabk	= MAX_KKINDEX*MAX_AAINDEX*64
		,MAX_kabbk  = MAX_KKINDEX*MAX_AAINDEX*64
		,MAX_kaaak	= MAX_KKINDEX*MAX_AAAINDEX
		,MAX_kapkb  = 24*64*64*64*64
		,MAX_kabkp  = 24*64*64*64*64
		,MAX_kabpk  = 24*64*64*64*64
		,MAX_kppka  = MAX_kppk*64
		,MAX_kappk  = MAX_kppk*64
		,MAX_kapkp  = MAX_kpkp*64
		,MAX_kaapk  = 24*MAX_AAINDEX*64*64
		,MAX_kaakp  = 24*MAX_AAINDEX*64*64
		,MAX_kppkp  = 24*MAX_PP48_INDEX*64*64
		,MAX_kpppk  = MAX_PPP48_INDEX*64*64	
};

#if defined(SHARED_forbuilding)
extern index_t 
biggest_memory_needed (void) {
    return MAX_kabkc;
}
#endif

/*--------------------------------*\
|
|
|		CACHE PROTOTYPES
|
|
*---------------------------------*/

#if !defined(SHARED_forbuilding)
mySHARED bool_t		get_dtm (tbkey_t key, unsigned side, index_t idx, dtm_t *out, bool_t probe_hard);
#endif

static bool_t	 	get_dtm_from_cache (tbkey_t key, unsigned side, index_t idx, dtm_t *out);


/*--------------------------------*\
|
|
|			INIT
|
|
*---------------------------------*/

static bool_t 	fd_init (struct filesopen *pfd);
static void		fd_done (struct filesopen *pfd);

static void		RAM_egtbfree (void);

/*--------------------------------------------------------------------------*/
#if !defined(SHARED_forbuilding)
mySHARED void   		egtb_freemem (int i);
#endif

mySHARED struct endgamekey egkey[] = {

{0, "kqk",  MAX_KXK,  1, kxk_indextopc,  kxk_pctoindex,  NULL ,  NULL   ,NULL ,0, 0 },
{1, "krk",  MAX_KXK,  1, kxk_indextopc,  kxk_pctoindex,  NULL ,  NULL   ,NULL ,0, 0 },
{2, "kbk",  MAX_KXK,  1, kxk_indextopc,  kxk_pctoindex,  NULL ,  NULL   ,NULL ,0, 0 },
{3, "knk",  MAX_KXK,  1, kxk_indextopc,  kxk_pctoindex,  NULL ,  NULL   ,NULL ,0, 0 },
{4, "kpk",  MAX_kpk,  24,kpk_indextopc,  kpk_pctoindex,  NULL ,  NULL   ,NULL ,0, 0 },
	/* 4 pieces */	
{5, "kqkq", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
{6, "kqkr", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
{7, "kqkb", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{8, "kqkn", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	

{9, "krkr", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{10,"krkb", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{11,"krkn", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{12,"kbkb", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{13,"kbkn", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{14,"knkn", MAX_kakb, 1, kakb_indextopc, kakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
	/**/		
{15,"kqqk", MAX_kaak, 1, kaak_indextopc, kaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
{16,"kqrk", MAX_kabk, 1, kabk_indextopc, kabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
{17,"kqbk", MAX_kabk, 1, kabk_indextopc, kabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{18,"kqnk", MAX_kabk, 1, kabk_indextopc, kabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	

{19,"krrk", MAX_kaak, 1, kaak_indextopc, kaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
{20,"krbk", MAX_kabk, 1, kabk_indextopc, kabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{21,"krnk", MAX_kabk, 1, kabk_indextopc, kabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{22,"kbbk", MAX_kaak, 1, kaak_indextopc, kaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
{23,"kbnk", MAX_kabk, 1, kabk_indextopc, kabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{24,"knnk", MAX_kaak, 1, kaak_indextopc, kaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },	
	/**/	
	/**/
{25,"kqkp", MAX_kakp, 24,kakp_indextopc, kakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{26,"krkp", MAX_kakp, 24,kakp_indextopc, kakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{27,"kbkp", MAX_kakp, 24,kakp_indextopc, kakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{28,"knkp", MAX_kakp, 24,kakp_indextopc, kakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
	/**/
{29,"kqpk", MAX_kapk, 24,kapk_indextopc, kapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{30,"krpk", MAX_kapk, 24,kapk_indextopc, kapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{31,"kbpk", MAX_kapk, 24,kapk_indextopc, kapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{32,"knpk", MAX_kapk, 24,kapk_indextopc, kapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
	/**/	
{33,"kppk", MAX_kppk, MAX_PPINDEX ,kppk_indextopc, kppk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
	/**/
{34,"kpkp", MAX_kpkp, MAX_PpINDEX ,kpkp_indextopc, kpkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
	/**/
	/**/
	/* 5 pieces */
{ 35,"kqqqk", MAX_kaaak, 1, kaaak_indextopc, kaaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 36,"kqqrk", MAX_kaabk, 1, kaabk_indextopc, kaabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 37,"kqqbk", MAX_kaabk, 1, kaabk_indextopc, kaabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 38,"kqqnk", MAX_kaabk, 1, kaabk_indextopc, kaabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 39,"kqrrk", MAX_kabbk, 1, kabbk_indextopc, kabbk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 40,"kqrbk", MAX_kabck, 1, kabck_indextopc, kabck_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 41,"kqrnk", MAX_kabck, 1, kabck_indextopc, kabck_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 42,"kqbbk", MAX_kabbk, 1, kabbk_indextopc, kabbk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 43,"kqbnk", MAX_kabck, 1, kabck_indextopc, kabck_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 44,"kqnnk", MAX_kabbk, 1, kabbk_indextopc, kabbk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 45,"krrrk", MAX_kaaak, 1, kaaak_indextopc, kaaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 46,"krrbk", MAX_kaabk, 1, kaabk_indextopc, kaabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 47,"krrnk", MAX_kaabk, 1, kaabk_indextopc, kaabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 48,"krbbk", MAX_kabbk, 1, kabbk_indextopc, kabbk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 49,"krbnk", MAX_kabck, 1, kabck_indextopc, kabck_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 50,"krnnk", MAX_kabbk, 1, kabbk_indextopc, kabbk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 51,"kbbbk", MAX_kaaak, 1, kaaak_indextopc, kaaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 52,"kbbnk", MAX_kaabk, 1, kaabk_indextopc, kaabk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 53,"kbnnk", MAX_kabbk, 1, kabbk_indextopc, kabbk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 54,"knnnk", MAX_kaaak, 1, kaaak_indextopc, kaaak_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 55,"kqqkq", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 56,"kqqkr", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 57,"kqqkb", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 58,"kqqkn", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 59,"kqrkq", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 60,"kqrkr", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 61,"kqrkb", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 62,"kqrkn", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 63,"kqbkq", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 64,"kqbkr", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 65,"kqbkb", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 66,"kqbkn", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 67,"kqnkq", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 68,"kqnkr", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 69,"kqnkb", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 70,"kqnkn", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 71,"krrkq", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 72,"krrkr", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 73,"krrkb", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 74,"krrkn", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 75,"krbkq", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 76,"krbkr", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 77,"krbkb", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 78,"krbkn", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 79,"krnkq", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 80,"krnkr", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 81,"krnkb", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 82,"krnkn", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 83,"kbbkq", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 84,"kbbkr", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 85,"kbbkb", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 86,"kbbkn", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 87,"kbnkq", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 88,"kbnkr", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 89,"kbnkb", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 90,"kbnkn", MAX_kabkc, 1, kabkc_indextopc, kabkc_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 91,"knnkq", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 92,"knnkr", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 93,"knnkb", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 94,"knnkn", MAX_kaakb, 1, kaakb_indextopc, kaakb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{ 95,"kqqpk", MAX_kaapk, 24, kaapk_indextopc, kaapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 96,"kqrpk", MAX_kabpk, 24, kabpk_indextopc, kabpk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 97,"kqbpk", MAX_kabpk, 24, kabpk_indextopc, kabpk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 98,"kqnpk", MAX_kabpk, 24, kabpk_indextopc, kabpk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{ 99,"krrpk", MAX_kaapk, 24, kaapk_indextopc, kaapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{100,"krbpk", MAX_kabpk, 24, kabpk_indextopc, kabpk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{101,"krnpk", MAX_kabpk, 24, kabpk_indextopc, kabpk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{102,"kbbpk", MAX_kaapk, 24, kaapk_indextopc, kaapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{103,"kbnpk", MAX_kabpk, 24, kabpk_indextopc, kabpk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{104,"knnpk", MAX_kaapk, 24, kaapk_indextopc, kaapk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{105,"kqppk", MAX_kappk, MAX_PPINDEX, kappk_indextopc, kappk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{106,"krppk", MAX_kappk, MAX_PPINDEX, kappk_indextopc, kappk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{107,"kbppk", MAX_kappk, MAX_PPINDEX, kappk_indextopc, kappk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{108,"knppk", MAX_kappk, MAX_PPINDEX, kappk_indextopc, kappk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{109,"kqpkq", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{110,"kqpkr", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{111,"kqpkb", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{112,"kqpkn", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{113,"krpkq", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{114,"krpkr", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{115,"krpkb", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{116,"krpkn", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{117,"kbpkq", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{118,"kbpkr", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{119,"kbpkb", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{120,"kbpkn", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{121,"knpkq", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{122,"knpkr", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{123,"knpkb", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{124,"knpkn", MAX_kapkb, 24, kapkb_indextopc, kapkb_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{125,"kppkq", MAX_kppka, MAX_PPINDEX, kppka_indextopc, kppka_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{126,"kppkr", MAX_kppka, MAX_PPINDEX, kppka_indextopc, kppka_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{127,"kppkb", MAX_kppka, MAX_PPINDEX, kppka_indextopc, kppka_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{128,"kppkn", MAX_kppka, MAX_PPINDEX, kppka_indextopc, kppka_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{129,"kqqkp", MAX_kaakp, 24, kaakp_indextopc, kaakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{130,"kqrkp", MAX_kabkp, 24, kabkp_indextopc, kabkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{131,"kqbkp", MAX_kabkp, 24, kabkp_indextopc, kabkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{132,"kqnkp", MAX_kabkp, 24, kabkp_indextopc, kabkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{133,"krrkp", MAX_kaakp, 24, kaakp_indextopc, kaakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{134,"krbkp", MAX_kabkp, 24, kabkp_indextopc, kabkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{135,"krnkp", MAX_kabkp, 24, kabkp_indextopc, kabkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{136,"kbbkp", MAX_kaakp, 24, kaakp_indextopc, kaakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{137,"kbnkp", MAX_kabkp, 24, kabkp_indextopc, kabkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{138,"knnkp", MAX_kaakp, 24, kaakp_indextopc, kaakp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{139,"kqpkp", MAX_kapkp, MAX_PpINDEX, kapkp_indextopc, kapkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{140,"krpkp", MAX_kapkp, MAX_PpINDEX, kapkp_indextopc, kapkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{141,"kbpkp", MAX_kapkp, MAX_PpINDEX, kapkp_indextopc, kapkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{142,"knpkp", MAX_kapkp, MAX_PpINDEX, kapkp_indextopc, kapkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{143,"kppkp", MAX_kppkp, 24*MAX_PP48_INDEX, kppkp_indextopc, kppkp_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },
{144,"kpppk", MAX_kpppk, MAX_PPP48_INDEX, kpppk_indextopc, kpppk_pctoindex, NULL ,  NULL   ,NULL ,0, 0 },

{MAX_EGKEYS, NULL,  0,        1, NULL,           NULL,           NULL,   NULL   ,NULL ,0 ,0}

};

static int eg_was_open[MAX_EGKEYS];

static uint64_t Bytes_read = 0;

/****************************************************************************\
|
|
|						PATH MANAGEMENT ZONE
|
|
\****************************************************************************/

#define MAXPATHLEN tb_MAXPATHLEN
#define MAX_GTBPATHS 10

static int  			Gtbpath_end_index = 0;
static const char **	Gtbpath = NULL;

/*---------------- EXTERNAL PATH MANAGEMENT --------------------------------*/

extern const char *tbpaths_getmain (void) {	return Gtbpath[0];}

extern const char **
tbpaths_init(void)
{
	const char **newps;
	newps = (const char **) malloc (sizeof (char *));
	if (newps != NULL) {
		newps[0] = NULL;
	}
	return newps;
}

static const char **
tbpaths_add_single(const char **ps, const char *newpath)
{
	size_t counter;
	const char **newps;
	size_t i, psize;
	char *ppath;

	if (NULL == ps)
		return NULL;

	psize = strlen(newpath) + 1;
	ppath = (char *) malloc (psize * sizeof (char));
	if (NULL == ppath)
		return ps; /* failed to incorporate a new path */
	for (i = 0; i < psize; i++) ppath[i] = newpath[i];

	for (counter = 0; ps[counter] != NULL; counter++)
		; 

	/* cast to deal with const poisoning */
	newps =	(const char **) realloc ((char **)ps, sizeof(char *) * (counter+2));
	if (newps != NULL) {
		newps [counter] = ppath;
		newps [counter+1] = NULL;
	}
	return newps;
}


extern const char **
tbpaths_add(const char **ps, const char *newpath)
{
	size_t i, psize;
	char *mpath;

	if (NULL == ps)
		return NULL;

	psize = strlen(newpath) + 1;
	mpath = (char *) malloc (psize * sizeof (char));
	if (NULL == mpath) {
		return ps; /* failed to incorporate a new path */
	}
	for (i = 0; i < psize; i++) mpath[i] = newpath[i];	

	for (i = 0; i < psize; i++) {
		if(';' == mpath[i])
			mpath[i] = '\0';	
	}

	for (i = 0;;) {
		while (i < psize && mpath[i] == '\0') i++;
		if (i >= psize) break;
		ps = tbpaths_add_single (ps, &mpath[i]);
		while (i < psize && mpath[i] != '\0') i++;
	}

	free(mpath);
	return ps;
}


extern const char **
tbpaths_done(const char **ps)
{
	int counter;
	void *q;	

	if (ps != NULL) {
		for (counter = 0; ps[counter] != NULL; counter++) {
			/* cast to deal with const poisoning */
			void *p = (void *) ps[counter];
			free(p);		
		} 	
		/* cast to deal with const poisoning */
		q = (void *) ps;
		free(q);
	}
	return NULL;
}

/*---------------- PATH INITIALIZATION ROUTINES ----------------------------*/

static void path_system_reset(void) {Gtbpath_end_index = 0;}

static bool_t
path_system_init (const char **path)
{
	size_t i;
	size_t sz;
	const char *x;
	bool_t ok = TRUE;
	path_system_reset();

	if (path == NULL) {
		return FALSE;
	}

	/* calculate needed size for Gtbpath */
	i = 0;
	do {
		x = path[i++];
	} while (x != NULL);
	sz = i; /* sz includes the NULL */


	Gtbpath = (const char **) malloc (sz * sizeof(char *));

	if (Gtbpath) {
		
		ok = TRUE;
		/* point to the same strings provided */
		Gtbpath_end_index = 0;
		for (i = 0; i < sz; i++) {
			Gtbpath[i] = path[i];
			Gtbpath_end_index++;
		}

	} else {
		ok = FALSE;
	}
	return ok;

}

static void
path_system_done (void)
{
	/* before we free Gtbpath, we have to deal with the
	"const poisoning" and cast it. free() does not accept
	const pointers */
	char **	p = (char **) Gtbpath;	
	/* clean up */
	if (p != NULL)
		free(p);
	return;
}


/****************************************************************************\
 *
 *
 *						General Initialization Zone
 *
 *
 ****************************************************************************/


#ifdef WDL_PROBE
static size_t 		wdl_cache_init (size_t cache_mem);
static void 		wdl_cache_flush (void);

static void			wdl_cache_reset_counters (void);
static void			wdl_cache_done (void);

static bool_t		get_WDL_from_cache (tbkey_t key, unsigned side, index_t idx, unsigned int *out);
static bool_t		wdl_preload_cache (tbkey_t key, unsigned side, index_t idx);
#endif

#ifdef GTB_SHARE
static void 	init_bettarr (void);
#endif

static void	eg_was_open_reset(void)
{
	int i;
	for (i = 0; i < MAX_EGKEYS; i++) {
		eg_was_open[i] = 0;
	}
}

static long unsigned int eg_was_open_count(void)
{
	long int i, x;
	for (i = 0, x = 0; i < MAX_EGKEYS; i++) {
		x += eg_was_open[i];
	}
	return (long unsigned) x;
}


enum  Sizes {INISIZE = 4096};
static char ini_str[INISIZE];
static void sjoin(char *s, const char *tail, size_t max) {strncat(s, tail, max - strlen(s) - 1);}

char *
tb_init (int verbosity, int decoding_sch, const char **paths)
{
	unsigned int zi;
	int paths_ok;
	char *ret_str;
	char localstr[256];

	assert(!TB_INITIALIZED);

	if (verbosity) {
		ini_str[0] = '\0';
		ret_str = ini_str;
	} else {
		ret_str = NULL;
	}

	paths_ok = path_system_init (paths);

	if (paths_ok && verbosity) { 
		int g;
		assert(Gtbpath!=NULL);
		sjoin(ini_str,"\nGTB PATHS\n",INISIZE);
		for (g = 0; Gtbpath[g] != NULL; g++) {
			const char *p = Gtbpath[g];
			if (0 == g) {
				sprintf (localstr,"  main: %s\n", p);
			} else {
				sprintf (localstr,"    #%d: %s\n", g, p);
			}
			sjoin(ini_str,localstr,INISIZE);
		}
	}

	if (!paths_ok && verbosity) { 
		sjoin (ini_str,"\nGTB PATHS not initialized\n",INISIZE);
	}

	if (!reach_was_initialized())
		reach_init();

	attack_maps_init (); /* external initialization */

	init_indexing(0 /* no verbosity */);	

	#ifdef GTB_SHARE
	init_bettarr();
	#endif

	if (!fd_init (&fd) && verbosity) {
		sjoin (ini_str,"  File Open Memory initialization = **FAILED**\n",INISIZE);
		return ret_str;
	}
	
	GTB_scheme = decoding_sch;
	Uncompressed = GTB_scheme == 0;

	if (GTB_scheme == 0) {
		Uncompressed = TRUE;
	}

	set_decoding_scheme(GTB_scheme);

	if (verbosity) {
		sjoin (ini_str,"\nGTB initialization\n",INISIZE);
		sprintf (localstr,"  Compression  Scheme = %d\n", GTB_scheme);
		sjoin (ini_str,localstr,INISIZE);
	}

	zi = zipinfo_init();

	TB_AVAILABILITY = zi;

	if (verbosity) {
		if (0 == zi) {
			sjoin (ini_str,"  Compression Indexes = **FAILED**\n",INISIZE);
		} else {
			int n, bit;

			n = 3; bit = 1;
			if (zi&(1u<<bit)) 
				sprintf (localstr,"  Compression Indexes (%d-pc) = PASSED\n",n);
			else
				sprintf (localstr,"  Compression Indexes (%d-pc) = **FAILED**\n",n);
			sjoin (ini_str,localstr,INISIZE);
			
			n = 4; bit = 3;
			if (zi&(1u<<bit))
				sprintf (localstr,"  Compression Indexes (%d-pc) = PASSED\n",n);
			else
				sprintf (localstr,"  Compression Indexes (%d-pc) = **FAILED**\n",n);
			sjoin (ini_str,localstr,INISIZE);

			n = 5; bit = 5;
			if (zi&(1u<<bit))
				sprintf (localstr,"  Compression Indexes (%d-pc) = PASSED\n",n);
			else
				sprintf (localstr,"  Compression Indexes (%d-pc) = **FAILED**\n",n);
			sjoin (ini_str,localstr,INISIZE);
		}
		sjoin (ini_str,"\n",INISIZE);
	}

	eg_was_open_reset();
	Bytes_read = 0;

	mythread_mutex_init (&Egtb_lock);

	TB_INITIALIZED = TRUE;

	return ret_str;
}

extern unsigned int
tb_availability(void)
{
	return TB_AVAILABILITY;
}

extern bool_t
tb_is_initialized (void)
{
	return TB_INITIALIZED;
}

extern void
tb_done (void)
{
	assert(TB_INITIALIZED);
	fd_done (&fd);
	RAM_egtbfree();
	zipinfo_done();
	path_system_done();
	mythread_mutex_destroy (&Egtb_lock);
	TB_INITIALIZED = FALSE;

	/*
		HERE, I should be free() the ini_str, but in
		the current implementation, it is static
		rather than dynamic.
	*/
	return;
}

char *
tb_restart(int verbosity, int decoding_sch, const char **paths)
{
	if (tb_is_initialized()) {
		tb_done();
	}
	return tb_init(verbosity, decoding_sch, paths);
}

/* whenever the program exits should release this memory */
static void
RAM_egtbfree (void)
{
	int i;
	for (i = 0; egkey[i].str != NULL; i++) {
		egtb_freemem (i);
	}
}

/*--------------------------------------------------------------------------*/

#ifdef GTB_SHARE
static void
init_bettarr (void)
{
/*
		iDRAW  = 0, iWMATE  = 1, iBMATE  = 2, iFORBID  = 3, 
		iDRAWt = 4, iWMATEt = 5, iBMATEt = 6, iUNKNOWN = 7
 */

	int temp[] = {
	/*White*/	
	/*iDRAW   vs*/
		1, 2, 1, 1,     2, 2, 2, 2,
	/*iWMATE  vs*/	
		1, 3, 1, 1,     1, 1, 1, 1,
	/*iBMATE  vs*/
		2, 2, 4, 1,     2, 2, 2, 2,
	/*iFORBID vs*/
		2, 2, 2, 2,     2, 2, 2, 2,
	
	/*iDRAWt  vs*/
		1, 2, 1, 1,     2, 2, 1, 2,
	/*iWMATEt vs*/	
		1, 2, 1, 1,     1, 3, 1, 1,
	/*iBMATEt vs*/
		1, 2, 1, 1,     2, 2, 4, 2,
	/*iUNKNOWN  */
		1, 2, 1, 1,     1, 2, 1, 2,

	/*Black*/
	/*iDRAW   vs*/
		1, 1, 2, 1,     2, 2, 2, 2,
	/*iWMATE  vs*/	
		2, 4, 2, 1,     2, 2, 2, 2,
	/*iBMATE  vs*/
		1, 1, 3, 1,     1, 1, 1, 1,
	/*iFORBID vs*/
		2, 2, 2, 2,     2, 2, 2, 2,
	
	/*iDRAWt  vs*/
		1, 1, 2, 1,     2, 1, 2, 2,
	/*iWMATEt vs*/	
		1, 1, 2, 1,     2, 4, 2, 2,
	/*iBMATEt vs*/
		1, 1, 2, 1,     1, 1, 3, 1,
	/*iUNKNOWN  */
		1, 1, 2, 1,     1, 1, 2, 2
	};

	int i, j, k, z;
	
	/* reset */
	z = 0;
	for (i = 0; i < 2; i++)
		for (j = 0; j < 8; j++) 
			for (k = 0; k < 8; k++)	
				bettarr [i][j][k] = temp[z++];

	return;
}
#endif

/*
|
|	Own File Descriptors
|
\*---------------------------------------------------------------------------*/

static bool_t
fd_init (struct filesopen *pfd)
{
	tbkey_t *p;
    int i, allowed;

	pfd->n = 0;

	allowed = mysys_fopen_max() - 5 /*stdin,stdout,sterr,stdlog,book*/;
	if (allowed < 4)
		GTB_MAXOPEN = 4;
	if (allowed > 32)
		GTB_MAXOPEN = 32;		

	p =	(tbkey_t *) malloc(sizeof(tbkey_t)*(size_t)GTB_MAXOPEN);

	if (p != NULL) {
		for (i = 0; i < GTB_MAXOPEN; i++) {
			p[i] = -1;	
		}
		pfd->key = p;
		return TRUE;
	} else {
		return FALSE;
	}
}

static void
fd_done (struct filesopen *pfd)
{
    int i;
	tbkey_t closingkey;
	FILE *finp;

	assert(pfd != NULL);

	for (i = 0; i < pfd->n; i++) {
		closingkey = pfd->key[i];
		finp = egkey [closingkey].fd;
		fclose (finp);
		egkey[closingkey].fd = NULL;
		pfd->key[i] = -1;	
	}
	pfd->n = 0;
	free(pfd->key);
}

/****************************************************************************\
|
|
|								PROBE ZONE
|
|
\****************************************************************************/

#if !defined(SHARED_forbuilding)

/* shared with building routines */
mySHARED void 			list_sq_copy 	(const SQUARE *a, SQUARE *b);
mySHARED void 			list_pc_copy 	(const SQ_CONTENT *a, SQ_CONTENT *b);
mySHARED dtm_t 			inv_dtm 		(dtm_t x);
mySHARED bool_t 		egtb_get_id  	(SQ_CONTENT *w, SQ_CONTENT *b, tbkey_t *id);
mySHARED void 			list_sq_flipNS 	(SQUARE *s);
mySHARED dtm_t 			adjust_up 		(dtm_t dist);
mySHARED dtm_t 			bestx 			(unsigned stm, dtm_t a, dtm_t b);
mySHARED void			sortlists 		(SQUARE *ws, SQ_CONTENT *wp);

mySHARED /*@NULL@*/ 	FILE * fd_openit(tbkey_t key);

mySHARED dtm_t 			dtm_unpack 	(unsigned stm, unsigned char packed);
mySHARED void  			unpackdist 	(dtm_t d, unsigned int *res, unsigned int *ply);
mySHARED dtm_t 			packdist 	(unsigned int inf, unsigned int ply);

mySHARED bool_t			fread_entry_packed 	(FILE *dest, unsigned side, dtm_t *px);
mySHARED bool_t			fpark_entry_packed  (FILE *finp, unsigned side, index_t max, index_t idx);
#endif

/* use only with probe */
static bool_t			egtb_get_dtm 	(tbkey_t k, unsigned stm, const SQUARE *wS, const SQUARE *bS, bool_t probe_hard, dtm_t *dtm);
static void				removepiece (SQUARE *ys, SQ_CONTENT *yp, int j);
static bool_t 			egtb_filepeek (tbkey_t key, unsigned side, index_t idx, dtm_t *out_dtm);


/*prototype*/
#ifdef WDL_PROBE
static bool_t
tb_probe_wdl
			(unsigned stm, 
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 bool_t probingtype,
			 /*@out@*/ unsigned *res);
#endif

static bool_t
tb_probe_	(unsigned stm, 
			 SQUARE epsq,
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 bool_t probingtype,
			 /*@out@*/ unsigned *res, 
			 /*@out@*/ unsigned *ply);


extern bool_t
tb_probe_soft
			(unsigned stm, 
			 SQUARE epsq,
			 unsigned castles,
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 /*@out@*/ unsigned *res, 
			 /*@out@*/ unsigned *ply)
{
	if (castles != 0) 
		return FALSE;
	return tb_probe_ (stm, epsq, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, FALSE, res, ply);
} 

extern bool_t
tb_probe_hard
			(unsigned stm, 
			 SQUARE epsq,
			 unsigned castles,
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 /*@out@*/ unsigned *res, 
			 /*@out@*/ unsigned *ply)
{
	if (castles != 0) 
		return FALSE;
	return tb_probe_ (stm, epsq, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, TRUE, res, ply);
} 

extern bool_t
tb_probe_WDL_soft
			(unsigned stm, 
			 SQUARE epsq,
			 unsigned castles,
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 /*@out@*/ unsigned *res)
{
	unsigned ply_n;
	unsigned *ply = &ply_n;
	if (castles != 0) 
		return FALSE;
	if (epsq != NOSQUARE)
		return tb_probe_ (stm, epsq, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, FALSE, res, ply);

	/* probe bitbase like, assuming no en passant */
	#ifdef WDL_PROBE
	return tb_probe_wdl    (stm, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, FALSE, res);
	#else
	return tb_probe_ (stm, epsq, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, FALSE, res, ply);
	#endif	
} 

extern bool_t
tb_probe_WDL_hard
			(unsigned stm, 
			 SQUARE epsq,
			 unsigned castles,
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 /*@out@*/ unsigned *res)
{
	unsigned ply_n;
	unsigned *ply = &ply_n;
	if (castles != 0) 
		return FALSE;
	if (epsq != NOSQUARE)
		return tb_probe_ (stm, epsq, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, TRUE, res, ply);

	/* probe bitbase like, assuming no en passant */
	#ifdef WDL_PROBE
	return tb_probe_wdl    (stm, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, TRUE, res);
	#else
	return tb_probe_ (stm, epsq, inp_wSQ, inp_bSQ, inp_wPC, inp_bPC, TRUE, res, ply);
	#endif	
} 


static bool_t
tb_probe_	(unsigned stm, 
			 SQUARE epsq,
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 bool_t probingtype,
			 /*@out@*/ unsigned *res, 
			 /*@out@*/ unsigned *ply)
{
	int i = 0, j = 0;
	tbkey_t id = -1;
	dtm_t dtm;

	SQUARE 		storage_ws [MAX_LISTSIZE], storage_bs [MAX_LISTSIZE];
	SQ_CONTENT  storage_wp [MAX_LISTSIZE], storage_bp [MAX_LISTSIZE];

	SQUARE     *ws = storage_ws;
	SQUARE     *bs = storage_bs;
	SQ_CONTENT *wp = storage_wp;
	SQ_CONTENT *bp = storage_bp;

	SQUARE     *xs;
	SQUARE     *ys;
	SQ_CONTENT *xp;
	SQ_CONTENT *yp;

	SQUARE 		tmp_ws [MAX_LISTSIZE], tmp_bs [MAX_LISTSIZE];
	SQ_CONTENT  tmp_wp [MAX_LISTSIZE], tmp_bp [MAX_LISTSIZE];

	SQUARE *temps;
	bool_t straight = FALSE;
	
	SQUARE capturer_a, capturer_b, xed = NOSQUARE;
	
	unsigned int plies;
	unsigned int inf;

	bool_t  okdtm  = TRUE;
	bool_t	okcall = TRUE;

	/************************************/

	assert (stm == WH || stm == BL);
	/*assert (inp_wPC[0] == KING && inp_bPC[0] == KING );*/
	assert ((epsq >> 3) == 2 || (epsq >> 3) == 5 || epsq == NOSQUARE);

	/* VALID ONLY FOR KK!! */
	if (inp_wPC[1] == NOPIECE && inp_bPC[1] == NOPIECE) {
		index_t dummy_i;
		bool_t b = kxk_pctoindex (inp_wSQ, inp_bSQ, &dummy_i);
		*res = b? iDRAW: iFORBID;
		*ply = 0;
		return TRUE;
	} 

	/* copy input */
	list_pc_copy (inp_wPC, wp);
	list_pc_copy (inp_bPC, bp);
	list_sq_copy (inp_wSQ, ws);
	list_sq_copy (inp_bSQ, bs);

	sortlists (ws, wp);
	sortlists (bs, bp);

	FOLLOW_label("EGTB_PROBE")

	if (egtb_get_id (wp, bp, &id)) {
		FOLLOW_LU("got ID",id)
		straight = TRUE;
	} else if (egtb_get_id (bp, wp, &id)) {
		FOLLOW_LU("rev ID",id)
		straight = FALSE;
		list_sq_flipNS (ws);
		list_sq_flipNS (bs);
        temps = ws;
        ws = bs;
        bs = temps;
		stm = Opp(stm);
		if (epsq != NOSQUARE) epsq ^= 070; 				/* added */
		{SQ_CONTENT *tempp = wp; wp = bp; bp = tempp;} 	/* added */
	} else {
		#if defined(DEBUG)
		printf("did not get id...\n");
		output_state (stm, ws, bs, wp, bp);		
		#endif
		unpackdist (iFORBID, res, ply);
		return FALSE;
	}

	/* store position... */
	list_pc_copy (wp, tmp_wp);
	list_pc_copy (bp, tmp_bp);
	list_sq_copy (ws, tmp_ws);
	list_sq_copy (bs, tmp_bs);

	/* x will be stm and y will be stw */
	if (stm == WH) {
        xs = ws;
        xp = wp;
        ys = bs;
        yp = bp;
    } else {
        xs = bs;
        xp = bp;
        ys = ws;
        yp = wp;
	}

	okdtm = egtb_get_dtm (id, stm, ws, bs, probingtype, &dtm);

	FOLLOW_LU("dtmok?",okdtm)
	FOLLOW_DTM("dtm", dtm)

	if (okdtm) {

		capturer_a = NOSQUARE;
		capturer_b = NOSQUARE;

		if (epsq != NOSQUARE) {
			/* captured pawn, trick: from epsquare to captured */		
			xed = epsq ^ (1<<3);

			/* find index captured (j) */
			for (j = 0; ys[j] != NOSQUARE; j++) {
				if (ys[j] == xed) break;
			}	

			/* try first possible ep capture */
			if (0 == (0x88 & (map88(xed) + 1))) 
				capturer_a = xed + 1;
			/* try second possible ep capture */
			if (0 == (0x88 & (map88(xed) - 1))) 
				capturer_b = xed - 1;

			if (ys[j] == xed) {
	
				/* find capturers (i) */
				for (i = 0; xs[i] != NOSQUARE && okcall; i++) {

					if (xp[i]==PAWN && (xs[i]==capturer_a || xs[i]==capturer_b)) {
						dtm_t epscore = iFORBID;

						/* execute capture */
						xs[i] = epsq;
						removepiece (ys, yp, j);
					
						okcall = tb_probe_ (Opp(stm), NOSQUARE, ws, bs, wp, bp, probingtype, &inf, &plies);
						
						if (okcall) {
							epscore = packdist (inf, plies); 					
							epscore = adjust_up (epscore);

							/* chooses to ep or not */
							dtm = bestx (stm, epscore, dtm);
						}
	
						/* restore position */
						list_pc_copy (tmp_wp, wp);
						list_pc_copy (tmp_bp, bp);

						list_sq_copy (tmp_ws, ws);
						list_sq_copy (tmp_bs, bs);					
					}
				}
			} 
		} /* ep */

		if (straight) {
			unpackdist (dtm, res, ply);
		} else {
			unpackdist (inv_dtm (dtm), res, ply);
		}	 
	} 

	if (!okdtm || !okcall) {
		unpackdist (iFORBID, res, ply);
	}
	
	return okdtm && okcall;
} 

#ifdef _MSC_VER
/* to silence warning for sprintf usage */
#pragma warning(disable:4996)
#endif

static bool_t
egtb_filepeek (tbkey_t key, unsigned side, index_t idx, dtm_t *out_dtm)
{
	FILE *finp;

#define USE_FD

	#if !defined(USE_FD)
	char buf[1024];
	char *filename = buf;
	#endif

	bool_t ok;
    dtm_t x=0;
	index_t maxindex  = egkey[key].maxindex;



	assert (Uncompressed);
	assert (side == WH || side == BL);
	assert (out_dtm != NULL);
	assert (idx >= 0);
	assert (key < MAX_EGKEYS);

		
	#if defined(USE_FD)
		if (NULL == (finp = egkey[key].fd) ) {
			if (NULL == (finp = fd_openit (key))) {
				return FALSE;
			}	
		}
	#else
		sprintf (buf, "%s.gtb", egkey[key].str);
		if (NULL == (finp = fopen (filename, "rb"))) {
			return FALSE;
		}	
	#endif

	ok = fpark_entry_packed (finp, side, maxindex, idx);
	ok = ok && fread_entry_packed (finp, side, &x);

	if (ok) {
		*out_dtm = x;		
	} else
		*out_dtm = iFORBID;

	#if !defined(USE_FD)
	fclose (finp);
	#endif

	return ok;
}

/* will get defined later */
static bool_t			dtm_cache_is_on (void);

static bool_t
egtb_get_dtm (tbkey_t k, unsigned stm, const SQUARE *wS, const SQUARE *bS, bool_t probe_hard_flag, dtm_t *dtm)
{
	bool_t idxavail;
	index_t idx;
	dtm_t *tab[2];
	bool_t (*pc2idx) (const SQUARE *, const SQUARE *, index_t *);

	FOLLOW_label("egtb_get_dtm --> starts")

	if (egkey[k].status == STATUS_MALLOC || egkey[k].status == STATUS_STATICRAM) {

		tab[WH] = egkey[k].egt_w;
		tab[BL] = egkey[k].egt_b;
		pc2idx  = egkey[k].pctoi;

		idxavail = pc2idx (wS, bS, &idx);

		FOLLOW_LU("indexavail (RAM)",idxavail)

		if (idxavail) {
			*dtm = tab[stm][idx];
		} else {
			*dtm = iFORBID;
		}

		return TRUE;

	} else if (egkey[k].status == STATUS_ABSENT) {

		pc2idx   = egkey[k].pctoi;
		idxavail = pc2idx (wS, bS, &idx);

		FOLLOW_LU("indexavail (HD)",idxavail)

		if (idxavail) {
			bool_t success;

			/* 
			|		LOCK 
			*-------------------------------*/
			mythread_mutex_lock (&Egtb_lock);	

			if (dtm_cache_is_on()) {

				success = get_dtm       (k, stm, idx, dtm, probe_hard_flag);

				FOLLOW_LU("get_dtm (succ)",success)
				FOLLOW_LU("get_dtm (dtm )",*dtm)

					#if defined(DEBUG)
					if (Uncompressed) {
						dtm_t 	dtm_temp;
						bool_t 	ok;
						bool_t 	success2;

						assert (decoding_scheme() == 0 && GTB_scheme == 0);

						success2 = egtb_filepeek (k, stm, idx, &dtm_temp);
						ok =  (success == success2) && (!success || *dtm == dtm_temp);
						if (!ok) {
							printf ("\nERROR\nsuccess1=%d sucess2=%d\n"
									"k=%d stm=%u idx=%d dtm_peek=%d dtm_cache=%d\n", 
									success, success2, k, stm, idx, dtm_temp, *dtm);
							fatal_error();
						}
					}
					#endif

			} else {	
				assert(Uncompressed);		
				if (probe_hard_flag && Uncompressed)
					success = egtb_filepeek (k, stm, idx, dtm);
				else
					success = FALSE;
			}

			mythread_mutex_unlock (&Egtb_lock);	
			/*------------------------------*\ 
			|		UNLOCK 
			*/
			

			if (success) {
				return TRUE;
			} else {
				if (probe_hard_flag) /* after probing hard and failing, no chance to succeed later */
					egkey[k].status = STATUS_REJECT;
				*dtm = iUNKNOWN;
				return FALSE;
			}

		} else {
			*dtm = iFORBID;
			return 	TRUE;
		}
		
	} else if (egkey[k].status == STATUS_REJECT) {

		FOLLOW_label("STATUS_REJECT")

		*dtm = iFORBID;
		return 	FALSE;
	} else {

		FOLLOW_label("STATUS_WRONG!")

		assert(0);
		*dtm = iFORBID;
		return 	FALSE;
	} 

} 

static void
removepiece (SQUARE *ys, SQ_CONTENT *yp, int j)
{
    int k;
	for (k = j; ys[k] != NOSQUARE; k++) {
		ys[k] = ys[k+1];
		yp[k] = yp[k+1];
	}
}

/* 
|
|	mySHARED by probe and build 
|
\*----------------------------------------------------*/

mySHARED /*@NULL@*/ FILE *
fd_openit (tbkey_t key)
{	
	int 			i;
	tbkey_t			closingkey;
	FILE *			finp = NULL;
	char	 		buf[4096];
	char *			filename = buf;
	int 			start; 
	int				end;
	int				pth;
	const char *	extension;

	assert (0 <= key && key < MAX_EGKEYS);
	assert (0 <= fd.n && fd.n <= GTB_MAXOPEN);

	/* test if I reach limit of files open */
	if (fd.n == GTB_MAXOPEN) {

		/* fclose the last accessed, first in the list */
		closingkey = fd.key[0];
		finp = egkey [closingkey].fd;
		assert (finp != NULL);
		fclose (finp);
		egkey[closingkey].fd = NULL;
		finp = NULL;

		for (i = 1; i < fd.n; i++) {
			fd.key[i-1] = fd.key[i]; 		
		}
		fd.key[--fd.n] = -1;
	} 

	assert (fd.n < GTB_MAXOPEN);

	/* set proper extensions to the File */
	if (Uncompressed) {
		assert (decoding_scheme() == 0 && GTB_scheme == 0);
		extension = ".gtb";
	} else {
		extension = Extension[decoding_scheme()];
	}

	/* Scan folders to find the File*/
	finp = NULL;

	start = egkey[key].pathn;
	end   = Gtbpath_end_index;

/*@@
printf ("start: %d\n",start);
printf ("===================Gtbpath[0]=%s\n",Gtbpath[0]);
*/
	for (pth = start; NULL == finp && pth < end && Gtbpath[pth] != NULL; pth++) {
		const char *path = Gtbpath[pth];
		size_t pl = strlen(path);
/*@@
printf ("path: %s\n",path);
*/
		if (pl == 0) {
				sprintf (buf, "%s%s%s", path, egkey[key].str, extension);
		} else {
			if (isfoldersep( path[pl-1] )) {
				sprintf (buf, "%s%s%s", path, egkey[key].str, extension);			
			} else {
				sprintf (buf, "%s%s%s%s", path, FOLDERSEP, egkey[key].str, extension);
			}
		}
/*printf ("try to open %s   --> ",filename);*/
		/* Finally found the file? */
		finp = fopen (filename, "rb");
/*printf ("%d\n",finp != NULL);*/
	}

	/* File was found and opened */
	if (NULL != finp) {
		fd.key [fd.n++] = key;
		egkey[key].fd = finp;
		egkey[key].pathn = pth; /* remember succesful path */
		eg_was_open[key] = 1;
		return finp;
	}

	start = 0;
	end   = egkey[key].pathn;
	for (pth = start; NULL == finp && pth < end && Gtbpath[pth] != NULL; pth++) {
		const char *path = Gtbpath[pth];
		size_t pl = strlen(path);

		if (pl == 0) {
				sprintf (buf, "%s%s%s", path, egkey[key].str, extension);
		} else {
			if (isfoldersep( path[pl-1] )) {
				sprintf (buf, "%s%s%s", path, egkey[key].str, extension);			
			} else {
				sprintf (buf, "%s%s%s%s", path, FOLDERSEP, egkey[key].str, extension);
			}
		}
/*printf ("try to open %s   --> ",filename);*/
		/* Finally found the file? */
		finp = fopen (filename, "rb");
/*printf ("%d\n",finp != NULL);*/
	}


	/* File was found and opened */
	if (NULL != finp) {
		fd.key [fd.n++] = key;
		egkey[key].fd = finp;
		egkey[key].pathn = pth; /* remember succesful path */
		eg_was_open[key] = 1;
	}

	return finp;
}

#ifdef _MSC_VER
/* to silence warning for sprintf usage */
#pragma warning(default:4996)
#endif

mySHARED void
sortlists (SQUARE *ws, SQ_CONTENT *wp)
{
	int i, j;
	SQUARE ts;
	SQ_CONTENT tp;
	/* input is sorted */
	for (i = 0; wp[i] != NOPIECE; i++) {
		for (j = (i+1); wp[j] != NOPIECE; j++) {	
			if (wp[j] > wp[i]) {
				tp = wp[i]; wp[i] = wp[j]; wp[j] = tp;		
				ts = ws[i]; ws[i] = ws[j]; ws[j] = ts;
			}			
		}	
	}
}

mySHARED void
list_sq_copy (const SQUARE *a, SQUARE *b)
{
	while (NOSQUARE != (*b++ = *a++))
		;
}

mySHARED void
list_pc_copy (const SQ_CONTENT *a, SQ_CONTENT *b)
{
	while (NOPIECE != (*b++ = *a++))
		;
}

mySHARED dtm_t
inv_dtm (dtm_t x)
{
	unsigned mat;
	assert ( (x & iUNKNBIT) == 0);

	if (x == iDRAW || x == iFORBID)
		return x;
	
	mat = (unsigned)x & 3u;
	if (mat == iWMATE)
		mat = iBMATE;
	else
		mat = iWMATE;

	x = (dtm_t) (((unsigned)x & ~3u) | mat);

	return x;
}

static const char pctoch[] = {'-','p','n','b','r','q','k'};

mySHARED bool_t
egtb_get_id (SQ_CONTENT *w, SQ_CONTENT *b, tbkey_t *id)
{

	char pcstr[2*MAX_LISTSIZE];
	SQ_CONTENT *s; 
	char *t;
	bool_t found;
	tbkey_t i;
	static tbkey_t cache_i = 0;
	
	assert (PAWN == 1 && KNIGHT == 2 && BISHOP == 3 && ROOK == 4 && QUEEN == 5 && KING == 6);

	t = pcstr; 

	s = w;
	while (NOPIECE != *s)
		*t++ = pctoch[*s++]; 		
	s = b;
	while (NOPIECE != *s)
		*t++ = pctoch[*s++]; 		

	*t = '\0';

	found = (0 == strcmp(pcstr, egkey[cache_i].str));
	if (found) {
		*id = cache_i;
		return found;		
	}
	
	for (i = 0, found = FALSE; !found && egkey[i].str != NULL; i++) {
		found = (0 == strcmp(pcstr, egkey[i].str));
	}
	if (found) {
		cache_i = *id = i - 1;
	}
	
	return found;
}

mySHARED void
list_sq_flipNS (SQUARE *s)
{
	while (*s != NOSQUARE) {
		*s ^= 070;
		s++;
	}
}

mySHARED void
unpackdist (dtm_t d, unsigned int *res, unsigned int *ply)
{
	*ply = (unsigned int)d >> PLYSHIFT;
	*res = d & INFOMASK;
}

mySHARED dtm_t
packdist (unsigned int inf, unsigned int ply)
{
	assert (inf <= INFOMASK);
	return (dtm_t) (inf | ply << PLYSHIFT);
}

mySHARED dtm_t
adjust_up (dtm_t dist)
{
	#if 0
	static const dtm_t adding[] = {	
		0, 1<<PLYSHIFT, 1<<PLYSHIFT, 0, 
		0, 1<<PLYSHIFT, 1<<PLYSHIFT, 0
	};
	dist += adding [dist&INFOMASK];
	return dist;
	#else			
	unsigned udist = (unsigned) dist;				
	switch (udist & INFOMASK) {
		case iWMATE:
		case iWMATEt:
		case iBMATE:
		case iBMATEt:
			udist += (1u << PLYSHIFT);
			break;
		default:			
			break;
	}
	return (dtm_t) udist;	
	#endif
}


mySHARED dtm_t
bestx (unsigned stm, dtm_t a, dtm_t b)
{
	unsigned int key;	
	static const unsigned int
	comparison [4] [4] = {
	 			/*draw, wmate, bmate, forbid*/
	/* draw  */ {0, 3, 0, 0},
	/* wmate */ {0, 1, 0, 0},
	/* bmate */ {3, 3, 2, 0},
	/* forbid*/ {3, 3, 3, 0}

	/* 0 = selectfirst   */
	/* 1 = selectlowest  */
	/* 2 = selecthighest */
	/* 3 = selectsecond  */
	};

	static const unsigned int xorkey [2] = {0, 3};	
	dtm_t retu[4];
	dtm_t ret = iFORBID;
	
	assert (stm == WH || stm == BL);
	assert ((a & iUNKNBIT) == 0 && (b & iUNKNBIT) == 0 );
	
	if (a == iFORBID)
		return b;
	if (b == iFORBID)
		return a;	

	retu[0] = a; /* first parameter */
	retu[1] = a; /* the lowest by default */
	retu[2] = b; /* highest by default */
	retu[3]	= b; /* second parameter */
	if (b < a) {
		retu[1] = b;
		retu[2] = a;		
	}
	
	key = comparison [a&3] [b&3] ^ xorkey[stm];
	ret = retu [key];
		
	return ret;
}


/*--------------------------------------------------------------------------*\
 |								PACKING ZONE
 *--------------------------------------------------------------------------*/

mySHARED dtm_t
dtm_unpack (unsigned stm, unsigned char packed)
{
	unsigned int info, plies, prefx, store, moves;
	dtm_t ret;
	unsigned int p = packed;

	if (iDRAW == p || iFORBID == p) {
		return (dtm_t) p;
	}
	
	info  = (unsigned int) p & 3;
	store = (unsigned int) p >> 2;

	if (WH == stm) {
		switch (info) {
			case iWMATE:		
						moves = store + 1;
						plies = moves * 2 - 1;
						prefx = info;
						break;				

			case iBMATE:
						moves = store;
						plies = moves * 2;
						prefx = info;
						break;

			case iDRAW:
						moves = store + 1 + 63;
						plies = moves * 2 - 1;
						prefx = iWMATE;						
						break;

			case iFORBID:

						moves = store + 63;
						plies = moves * 2;
						prefx = iBMATE;
						break;
			default:	
            plies = 0;
            prefx = 0;
            assert(0);
						break;

		}
		ret = (dtm_t) (prefx | (plies << 3));
	} else {
		switch (info) {
			case iBMATE:		
						moves = store + 1;
						plies = moves * 2 - 1;
						prefx = info;
						break;				

			case iWMATE:
						moves = store;
						plies = moves * 2;
						prefx = info;
						break;

			case iDRAW:

						if (store == 63) {
						/* 	exception: no position in the 5-man 
							TBs needs to store 63 for iBMATE 
							it is then used to indicate iWMATE 
							when just overflows */
							store++;

							moves = store + 63;
							plies = moves * 2;
							prefx = iWMATE;	

							break;
						}
	
						moves = store + 1 + 63;
						plies = moves * 2 - 1;
						prefx = iBMATE;						
						break;

			case iFORBID:

						moves = store + 63;
						plies = moves * 2;
						prefx = iWMATE;
						break;
			default:	
            plies = 0;
            prefx = 0;
            assert(0);
						break;

		}
		ret = (dtm_t) (prefx | (plies << 3));
	}
	return ret;	
}


/*
static bool_t fwrite_entry_packed (FILE *dest, unsigned side, dtm_t x);
*/


mySHARED bool_t
fread_entry_packed (FILE *finp, unsigned side, dtm_t *px)
{
	unsigned char p[SLOTSIZE];
	bool_t ok = (SLOTSIZE == fread (p, sizeof(unsigned char), SLOTSIZE, finp));
	if (ok) {
		*px = dtm_unpack (side, p[0]);
	}
	return ok;
}


mySHARED bool_t
fpark_entry_packed  (FILE *finp, unsigned side, index_t max, index_t idx)
{
	bool_t ok;
	index_t i;
	long int fseek_i;
	index_t sz = (index_t) sizeof(unsigned char);	

	assert (side == WH || side == BL);
	assert (finp != NULL);
	assert (idx >= 0);
	i = ((index_t)side * max + idx) * sz;
	fseek_i = (long int) i;
	assert (fseek_i >= 0);
	ok = (0 == fseek (finp, fseek_i, SEEK_SET));
	return ok;
}

/*----------------------------------------------------*\ 
|
|	shared by probe and build 
|
\*/


/*---------------------------------------------------------------------*\
|			WDL CACHE Implementation  ZONE
\*---------------------------------------------------------------------*/

#define WDL_entries_per_unit 4
#define WDL_entry_mask     3
static size_t		WDL_units_per_block = 0;

static bool_t		WDL_CACHE_INITIALIZED = FALSE;

typedef unsigned char unit_t; /* block unit */

typedef struct wdl_block 	wdl_block_t;

struct wdl_block {
	tbkey_t			key;
	unsigned		side;
	index_t 		offset;
	unit_t			*p_arr;
	wdl_block_t		*prev;
	wdl_block_t		*next;
};

struct WDL_CACHE {
	/* defined at init */
	bool_t			cached;
	size_t			max_blocks;
	size_t 			entries_per_block;
	unit_t		 *	buffer;

	/* flushables */
	wdl_block_t	*	top;
	wdl_block_t *	bot;
	size_t			n;
	wdl_block_t *	blocks; /* was entry */

	/* counters */
	uint64_t		hard;
	uint64_t		soft;
	uint64_t		hardmisses;
	uint64_t		hits;
	uint64_t		softmisses;
	uint64_t 		comparisons;
};

struct WDL_CACHE 	wdl_cache = {FALSE,0,0,NULL,
								 NULL,NULL,0,NULL,
								 0,0,0,0,0,0};


/*---------------------------------------------------------------------*\
|			DTM CACHE Implementation  ZONE
\*---------------------------------------------------------------------*/

struct dtm_block;

typedef struct dtm_block dtm_block_t;

struct dtm_block {
	tbkey_t			key;
	unsigned		side;
	index_t 		offset;
	dtm_t			*p_arr;
	dtm_block_t		*prev;
	dtm_block_t		*next;
};

struct cache_table {
	/* defined at init */
	bool_t			cached;
	size_t			max_blocks;
	size_t 			entries_per_block;
	dtm_t *			buffer;

	/* flushables */
	dtm_block_t	*	top;
	dtm_block_t *	bot;
	size_t			n;
	dtm_block_t *	entry;

	/* counters */
	uint64_t		hard;
	uint64_t		soft;
	uint64_t		hardmisses;
	uint64_t		hits;
	uint64_t		softmisses;
	unsigned long	comparisons;
};

struct cache_table 	dtm_cache = {FALSE,0,0,NULL,
								NULL,NULL,0,NULL,
								0,0,0,0,0,0};

struct general_counters {
	/* counters */
	uint64_t		hits;
	uint64_t		miss;
};

static struct general_counters Drive = {0,0};


static void 		split_index (size_t entries_per_block, index_t i, index_t *o, index_t *r);
static dtm_block_t *point_block_to_replace (void);
static bool_t 		preload_cache (tbkey_t key, unsigned side, index_t idx);
static void			movetotop (dtm_block_t *t);

/*--cache prototypes--------------------------------------------------------*/

/*- WDL --------------------------------------------------------------------*/
#ifdef WDL_PROBE
static unsigned int		wdl_extract (unit_t *uarr, index_t x);
static wdl_block_t *	wdl_point_block_to_replace (void);
static void				wdl_movetotop (wdl_block_t *t);

#if 0
static bool_t			wdl_cache_init (size_t cache_mem);
static void				wdl_cache_flush (void);
static bool_t			get_WDL (tbkey_t key, unsigned side, index_t idx, unsigned int *info_out, bool_t probe_hard_flag);
#endif

static bool_t			wdl_cache_is_on (void);
static void				wdl_cache_reset_counters (void);
static void				wdl_cache_done (void);

static wdl_block_t *	wdl_point_block_to_replace (void);
static bool_t			get_WDL_from_cache (tbkey_t key, unsigned side, index_t idx, unsigned int *out);
static void				wdl_movetotop (wdl_block_t *t);
static bool_t			wdl_preload_cache (tbkey_t key, unsigned side, index_t idx);
#endif
/*--------------------------------------------------------------------------*/
/*- DTM --------------------------------------------------------------------*/
static bool_t			dtm_cache_is_on (void);
static void				dtm_cache_reset_counters (void);
static void				dtm_cache_done (void);

static size_t			dtm_cache_init (size_t cache_mem);
static void				dtm_cache_flush (void);
/*--------------------------------------------------------------------------*/

static bool_t
dtm_cache_is_on (void)
{
	return dtm_cache.cached;
}

static void
dtm_cache_reset_counters (void)
{
	dtm_cache.hard = 0;
	dtm_cache.soft = 0;
	dtm_cache.hardmisses = 0;
	dtm_cache.hits = 0;
	dtm_cache.softmisses = 0;
	dtm_cache.comparisons = 0;
	return;
}


static size_t
dtm_cache_init (size_t cache_mem)
{
	unsigned int 	i;
	dtm_block_t 	*p;
	size_t 			entries_per_block;
	size_t 			max_blocks;
	size_t 			block_mem; 

	if (DTM_CACHE_INITIALIZED)
		dtm_cache_done();

	entries_per_block 	= 16 * 1024;  /* fixed, needed for the compression schemes */

	block_mem 			= entries_per_block * sizeof(dtm_t);

	max_blocks 			= cache_mem / block_mem;
	if (!Uncompressed && 1 > max_blocks)
		max_blocks = 1; 
	cache_mem 			= max_blocks * block_mem;


	dtm_cache_reset_counters ();

	dtm_cache.entries_per_block	= entries_per_block;
	dtm_cache.max_blocks 		= max_blocks;
	dtm_cache.cached 			= TRUE;
	dtm_cache.top 				= NULL;
	dtm_cache.bot 				= NULL;
	dtm_cache.n 				= 0;

	if (0 == cache_mem || NULL == (dtm_cache.buffer = (dtm_t *)  malloc (cache_mem))) {
		dtm_cache.cached = FALSE;
		dtm_cache.buffer = NULL;
		dtm_cache.entry = NULL;
		return 0;
	}

	if (0 == max_blocks|| NULL == (dtm_cache.entry  = (dtm_block_t *) malloc (max_blocks * sizeof(dtm_block_t)))) {
		dtm_cache.cached = FALSE;
		dtm_cache.entry = NULL;
		free (dtm_cache.buffer);
		dtm_cache.buffer = NULL;
		return 0;
	}
	
	for (i = 0; i < max_blocks; i++) {
		p = &dtm_cache.entry[i];
		p->key  	= -1;
		p->side 	= gtbNOSIDE;
		p->offset 	= gtbNOINDEX;
		p->p_arr 	= dtm_cache.buffer + i * entries_per_block;
		p->prev 	= NULL;
		p->next 	= NULL;
	}

	DTM_CACHE_INITIALIZED = TRUE;

	return cache_mem;
}


static void
dtm_cache_done (void)
{
	assert(DTM_CACHE_INITIALIZED);

	dtm_cache.cached = FALSE;
	dtm_cache.hard = 0;
	dtm_cache.soft = 0;
	dtm_cache.hardmisses = 0;
	dtm_cache.hits = 0;
	dtm_cache.softmisses = 0;
	dtm_cache.comparisons = 0;
	dtm_cache.max_blocks = 0;
	dtm_cache.entries_per_block = 0;

	dtm_cache.top = NULL;
	dtm_cache.bot = NULL;
	dtm_cache.n = 0;

	if (dtm_cache.buffer != NULL)
		free (dtm_cache.buffer);
	dtm_cache.buffer = NULL;

	if (dtm_cache.entry != NULL)
		free (dtm_cache.entry);
	dtm_cache.entry = NULL;

	DTM_CACHE_INITIALIZED = FALSE;

	return;
}

static void
dtm_cache_flush (void)
{
	unsigned int 	i;
	dtm_block_t 	*p;
	size_t entries_per_block = dtm_cache.entries_per_block;
	size_t max_blocks = dtm_cache.max_blocks;

	dtm_cache.top 				= NULL;
	dtm_cache.bot 				= NULL;
	dtm_cache.n 				= 0;
	
	for (i = 0; i < max_blocks; i++) {
		p = &dtm_cache.entry[i];
		p->key  	= -1;
		p->side 	= gtbNOSIDE;
		p->offset 	= gtbNOINDEX;
		p->p_arr 	= dtm_cache.buffer + i * entries_per_block;
		p->prev 	= NULL;
		p->next 	= NULL;
	}
	dtm_cache_reset_counters ();
	return;
}


/*---- end tbcache zone ----------------------------------------------------------------------*/

extern bool_t
tbcache_is_on (void)
{
	return dtm_cache_is_on() || wdl_cache_is_on();
}


/* STATISTICS OUTPUT */

extern void 
tbstats_get (struct TB_STATS *x)
{
	long unsigned mask = 0xfffffffflu;
	uint64_t memory_hits, total_hits;


	/*
	|	WDL CACHE
	\*---------------------------------------------------*/

	x->wdl_easy_hits[0] = (long unsigned)(wdl_cache.hits & mask);
	x->wdl_easy_hits[1] = (long unsigned)(wdl_cache.hits >> 32);

	x->wdl_hard_prob[0] = (long unsigned)(wdl_cache.hard & mask);
	x->wdl_hard_prob[1] = (long unsigned)(wdl_cache.hard >> 32);

	x->wdl_soft_prob[0] = (long unsigned)(wdl_cache.soft & mask);
	x->wdl_soft_prob[1] = (long unsigned)(wdl_cache.soft >> 32);

	x->wdl_cachesize    = WDL_cache_size;

	/* occupancy */
	x->wdl_occupancy = wdl_cache.max_blocks==0? 0:(double)100.0*(double)wdl_cache.n/(double)wdl_cache.max_blocks;

	/*
	|	DTM CACHE
	\*---------------------------------------------------*/

	x->dtm_easy_hits[0] = (long unsigned)(dtm_cache.hits & mask);
	x->dtm_easy_hits[1] = (long unsigned)(dtm_cache.hits >> 32);

	x->dtm_hard_prob[0] = (long unsigned)(dtm_cache.hard & mask);
	x->dtm_hard_prob[1] = (long unsigned)(dtm_cache.hard >> 32);

	x->dtm_soft_prob[0] = (long unsigned)(dtm_cache.soft & mask);
	x->dtm_soft_prob[1] = (long unsigned)(dtm_cache.soft >> 32);

	x->dtm_cachesize    = DTM_cache_size;

	/* occupancy */
	x->dtm_occupancy = dtm_cache.max_blocks==0? 0:(double)100.0*(double)dtm_cache.n/(double)dtm_cache.max_blocks;

	/*
	|	GENERAL
	\*---------------------------------------------------*/

	/* memory */
	memory_hits = wdl_cache.hits + dtm_cache.hits;
	x->memory_hits[0] = (long unsigned)(memory_hits & mask);
	x->memory_hits[1] = (long unsigned)(memory_hits >> 32);

	/* hard drive */
	x->drive_hits[0] = (long unsigned)(Drive.hits & mask);
	x->drive_hits[1] = (long unsigned)(Drive.hits >> 32);

	x->drive_miss[0] = (long unsigned)(Drive.miss & mask);
	x->drive_miss[1] = (long unsigned)(Drive.miss >> 32);

	x->bytes_read[0] = (long unsigned)(Bytes_read & mask);
	x->bytes_read[1] = (long unsigned)(Bytes_read >> 32);

	x->files_opened = eg_was_open_count();

	/* total */
	total_hits = memory_hits + Drive.hits;
	x->total_hits[0] = (long unsigned)(total_hits & mask);
	x->total_hits[1] = (long unsigned)(total_hits >> 32);

	/* efficiency */
	{ uint64_t denominator = memory_hits + Drive.hits + Drive.miss;
	x->memory_efficiency = 0==denominator? 0: 100.0 * (double)(memory_hits) / (double)(denominator);
	}
}


extern bool_t
tbcache_init (size_t cache_mem, int wdl_fraction)
{
	assert (wdl_fraction <= WDL_FRACTION_MAX && wdl_fraction >= 0);

	/* defensive against input */
	if (wdl_fraction > WDL_FRACTION_MAX) wdl_fraction = WDL_FRACTION_MAX;
	if (wdl_fraction <                0) wdl_fraction = 0;
	WDL_FRACTION = wdl_fraction;
	
	DTM_cache_size = (cache_mem/(size_t)WDL_FRACTION_MAX)*(size_t)(WDL_FRACTION_MAX-WDL_FRACTION);
	WDL_cache_size = (cache_mem/(size_t)WDL_FRACTION_MAX)*(size_t)     				WDL_FRACTION ;

	#ifdef WDL_PROBE
	/* returns the actual memory allocated */
	DTM_cache_size = dtm_cache_init (DTM_cache_size);
	WDL_cache_size = wdl_cache_init (WDL_cache_size);
	#else
	/* returns the actual memory allocated */
	DTM_cache_size = dtm_cache_init (DTM_cache_size);
	#endif
	tbstats_reset ();
	return TRUE;
}

extern bool_t
tbcache_restart (size_t cache_mem, int wdl_fraction)
{
	return tbcache_init (cache_mem, wdl_fraction);
}

extern void
tbcache_done (void)
{
	dtm_cache_done();
	#ifdef WDL_PROBE
	wdl_cache_done();
	#endif
	tbstats_reset ();
	return;
}

extern void
tbcache_flush (void)
{
	dtm_cache_flush();
	#ifdef WDL_PROBE
	wdl_cache_flush();
	#endif
	tbstats_reset ();
	return;
}

extern void	
tbstats_reset (void)
{
	dtm_cache_reset_counters ();
	#ifdef WDL_PROBE
	wdl_cache_reset_counters ();
	#endif
	eg_was_open_reset();
	Drive.hits = 0;
	Drive.miss = 0;
	return;
}

static dtm_block_t	*
dtm_cache_pointblock (tbkey_t key, unsigned side, index_t idx)
{
	index_t 		offset;
	index_t			remainder;
	dtm_block_t	*	p;
	dtm_block_t	*	ret;

	if (!dtm_cache_is_on())
		return NULL;

	split_index (dtm_cache.entries_per_block, idx, &offset, &remainder); 

	ret   = NULL;

	for (p = dtm_cache.top; p != NULL; p = p->prev) {

		dtm_cache.comparisons++;

		if (key == p->key && side == p->side && offset  == p->offset) {
			ret = p;
			break;
		}
	}

	FOLLOW_LU("point_to_dtm_block ok?",(ret!=NULL))

	return ret;
}

/****************************************************************************\
|
|
|			WRAPPERS for ENCODING/DECODING FUNCTIONS ZONE
|
|
\****************************************************************************/

#include "gtb-dec.h"

/*
|
|	PRE LOAD CACHE AND DEPENDENT FUNCTIONS 
|
\*--------------------------------------------------------------------------*/

struct ZIPINFO {
	index_t 	extraoffset;
	index_t 	totalblocks;
	index_t *	blockindex;
};

struct ZIPINFO Zipinfo[MAX_EGKEYS];

static index_t 	egtb_block_getnumber 		(tbkey_t key, unsigned side, index_t idx);
static index_t 	egtb_block_getsize 			(tbkey_t key, index_t idx);
static index_t 	egtb_block_getsize_zipped 	(tbkey_t key, index_t block );
static  bool_t 	egtb_block_park  			(tbkey_t key, index_t block);
static  bool_t 	egtb_block_read 			(tbkey_t key, index_t len, unsigned char *buffer); 
static  bool_t 	egtb_block_decode 			(tbkey_t key, index_t z, unsigned char *bz, index_t n, unsigned char *bp);
static  bool_t 	egtb_block_unpack 			(unsigned side, index_t n, const unsigned char *bp, dtm_t *out);
static  bool_t 	egtb_file_beready 			(tbkey_t key);
static  bool_t 	egtb_loadindexes 			(tbkey_t key);
static index_t 	egtb_block_uncompressed_to_index (tbkey_t key, index_t b);
static  bool_t 	fread32 					(FILE *f, /*@out@*/ unsigned long int *y);


static unsigned int
zipinfo_init (void)
{
	int i, start, end;
	unsigned ret;
	bool_t ok, complet[8] = {0,0,0,0,0,0,0,0};
	bool_t pa, partial[8] = {0,0,0,0,0,0,0,0};
	unsigned int z;
	int x, j;

	/* reset all values */
	for (i = 0; i < MAX_EGKEYS; i++) {
		Zipinfo[i].blockindex = NULL;
	 	Zipinfo[i].extraoffset = 0;
	 	Zipinfo[i].totalblocks = 0;
	}

	/* load all values */
	start = 0;
	end   = 5;
	x	  = 3;
	for (i = start, ok = TRUE, pa = FALSE; i < end; i++) {
		ok = NULL != fd_openit(i);
		pa = pa || ok;
		ok = ok && egtb_loadindexes (i);
	}
	complet[x] = ok;
	partial[x] = pa;

	start = 5;
	end   = 35;
	x	  = 4;
	for (i = start, ok = TRUE, pa = FALSE; i < end; i++) {
		ok = NULL != fd_openit(i);
		pa = pa || ok;
		ok = ok && egtb_loadindexes (i);
	}
	complet[x] = ok;
	partial[x] = pa;

	start = 35;
	end   = MAX_EGKEYS;
	x	  = 5;
	for (i = start, ok = TRUE, pa = FALSE; i < end; i++) {
		ok = NULL != fd_openit(i);
		pa = pa || ok;
		ok = ok && egtb_loadindexes (i);
	}
	complet[x] = ok;
	partial[x] = pa;


	for (j = 0, z = 0, x = 3; x < 8; x++) {
		if (partial[x]) z |= 1u << j; 
		j++;
		if (complet[x]) z |= 1u << j;
		j++;
	}

	ret = z;

	return ret;
}

static void
zipinfo_done (void)
{
	int i;
	bool_t ok;
	for (i = 0, ok = TRUE; ok && i < MAX_EGKEYS; i++) {
		if (Zipinfo[i].blockindex != NULL) {
			free(Zipinfo[i].blockindex);
			Zipinfo[i].blockindex = NULL;
		 	Zipinfo[i].extraoffset = 0;
		 	Zipinfo[i].totalblocks = 0;
		}
	}
	return;
}

static size_t
zipinfo_memory_allocated (void)
{
	int i;
	index_t accum_blocks = 0;
	for (i = 0; i < MAX_EGKEYS; i++) {
		if (Zipinfo[i].blockindex != NULL) {
		 	accum_blocks += Zipinfo[i].totalblocks;
		}
	}
	return (size_t)accum_blocks * sizeof(index_t);
}

extern size_t
tb_indexmemory (void)
{
	return zipinfo_memory_allocated ();
}

static bool_t
fread32 (FILE *f, /*@out@*/ unsigned long int *y)
{
	enum SIZE {SZ = 4};
	int i;
	unsigned long int x;
	unsigned char p[SZ];
	bool_t ok;

	ok = (SZ == fread (p, sizeof(unsigned char), SZ, f));

	if (ok) {
		for (x = 0, i = 0; i < SZ; i++) {
			x |= (unsigned long int)p[i] << (i*8);
		}
		*y = x;
	}
	return ok;
}

static bool_t
egtb_loadindexes (tbkey_t key)
{

	unsigned long int blocksize = 1;
	unsigned long int tailblocksize1 = 0;
	unsigned long int tailblocksize2 = 0;
    unsigned long int offset = 0;
	unsigned long int dummy;
	unsigned long int i;
	unsigned long int blocks;
	unsigned long int n_idx;
	unsigned long int idx = 0; /* to silence warning "may be used uninitialized..." */
	index_t	*p;

	bool_t ok;

	FILE *f;

	if (Uncompressed) {
		assert (decoding_scheme() == 0 && GTB_scheme == 0);	
		return TRUE; /* no need to load indexes */
	}
	if (Zipinfo[key].blockindex != NULL)
		return TRUE; /* indexes must have been loaded already */

	if (NULL == (f = egkey[key].fd))
		return FALSE; /* file was no open */

	/* Get Reserved bytes, blocksize, offset */
	ok = (0 == fseek (f, 0, SEEK_SET)) &&
	fread32 (f, &dummy) &&	
	fread32 (f, &dummy) &&
	fread32 (f, &blocksize) &&
	fread32 (f, &dummy) &&
	fread32 (f, &tailblocksize1) &&
	fread32 (f, &dummy) &&
	fread32 (f, &tailblocksize2) &&
	fread32 (f, &dummy) &&
	fread32 (f, &offset) &&
	fread32 (f, &dummy);

	blocks = (offset - 40)/4 -1;
	n_idx = blocks + 1;

	p = NULL;

	ok = ok && NULL != (p = (index_t *)malloc (n_idx * sizeof(index_t)));

	/* Input of Indexes */
	for (i = 0; ok && i < n_idx; i++) {
		ok = fread32 (f, &idx); /* idx will be used if ok is true, otherwise, it is discarded */
		p[i] = (index_t)idx; /* reads a 32 bit int, and converts it to index_t */ assert (sizeof(index_t) >= 4);
	}

	if (ok) {
		Zipinfo[key].extraoffset = 0;	
		assert (n_idx <= MAXINDEX_T);
		Zipinfo[key].totalblocks = (index_t) n_idx; 
		Zipinfo[key].blockindex  = p;
	}	

	if (!ok && p != NULL) {
		free(p);
	}

	return ok;
}

static index_t
egtb_block_uncompressed_to_index (tbkey_t key, index_t b)
{
	index_t max;
	index_t blocks_per_side;
	index_t idx;

	max = egkey[key].maxindex;
	blocks_per_side = 1 + (max-1) / (index_t)dtm_cache.entries_per_block;

	if (b < blocks_per_side) {
		idx = 0;
	} else {
		b -= blocks_per_side;
		idx = max;
	}
	idx += b * (index_t)dtm_cache.entries_per_block;
	return idx;
}


static index_t
egtb_block_getnumber (tbkey_t key, unsigned side, index_t idx)
{
	index_t blocks_per_side;
	index_t block_in_side;
	index_t max = egkey[key].maxindex;

	blocks_per_side = 1 + (max-1) / (index_t)dtm_cache.entries_per_block;
	block_in_side   = idx         / (index_t)dtm_cache.entries_per_block;

	return (index_t)side * blocks_per_side + block_in_side; /* block */
}


static index_t 
egtb_block_getsize (tbkey_t key, index_t idx)
{
	index_t blocksz = (index_t) dtm_cache.entries_per_block;
	index_t maxindex  = egkey[key].maxindex;
	index_t block, offset, x; 

	assert (dtm_cache.entries_per_block <= MAXINDEX_T);
	assert (0 <= idx && idx < maxindex);
	assert (key < MAX_EGKEYS);

	block = idx / blocksz;
	offset = block * blocksz;

	/* 
	|	adjust block size in case that this is the last block 
	|	and is shorter than "blocksz" 
	*/
	if ( (offset + blocksz) > maxindex) 
		x = maxindex - offset; /* last block size */
	else
		x = blocksz; /* size of a normal block */
	
	return x;
}

static index_t 
egtb_block_getsize_zipped (tbkey_t key, index_t block )
{
	index_t i, j;
	assert (Zipinfo[key].blockindex != NULL);
	i = Zipinfo[key].blockindex[block];
	j = Zipinfo[key].blockindex[block+1];	
	return j - i;
}


static bool_t
egtb_file_beready (tbkey_t key)
{
	bool_t success;
	assert (key < MAX_EGKEYS);
	success = 	(NULL != egkey[key].fd) ||
				(NULL != fd_openit(key) && egtb_loadindexes (key));
	return success; 
}


static bool_t
egtb_block_park  (tbkey_t key, index_t block)
{
	index_t i;
	long fseek_i;
	assert (egkey[key].fd != NULL);

	if (Uncompressed) {
		assert (decoding_scheme() == 0 && GTB_scheme == 0);	
		i = egtb_block_uncompressed_to_index (key, block);
	} else {
		assert (Zipinfo[key].blockindex != NULL);
		i  = Zipinfo[key].blockindex[block];
		i += Zipinfo[key].extraoffset;
	}

	fseek_i = (long) i;
	assert (fseek_i >= 0);
	return 0 == fseek (egkey[key].fd, fseek_i, SEEK_SET);
}


static bool_t
egtb_block_read (tbkey_t key, index_t len, unsigned char *buffer) 
{
	assert (egkey[key].fd != NULL);
	assert (sizeof(size_t) >= sizeof(len));
	return ((size_t)len == fread (buffer, sizeof (unsigned char), (size_t)len, egkey[key].fd));	
}

tbkey_t TB_PROBE_indexing_dummy;

static bool_t
egtb_block_decode (tbkey_t key, index_t z, unsigned char *bz, index_t n, unsigned char *bp)
/* bz:buffer zipped to bp:buffer packed */
{
	size_t zz = (size_t) z;
	size_t nn = (size_t) n;
	TB_PROBE_indexing_dummy = key; /* to silence compiler */
	assert (sizeof(size_t) >= sizeof(n));
	assert (sizeof(size_t) >= sizeof(z));
	return decode (zz-1, bz+1, nn, bp);
}

static bool_t
egtb_block_unpack (unsigned side, index_t n, const unsigned char *bp, dtm_t *out)
/* bp:buffer packed to out:distance to mate buffer */
{
	index_t i;
	for (i = 0; i < n; i++) {
		*out++ = dtm_unpack (side, bp[i]);		
	}	
	return TRUE;
}

static bool_t
preload_cache (tbkey_t key, unsigned side, index_t idx)
/* output to the least used block of the cache */
{
	dtm_block_t 	*pblock;
	dtm_t 			*p;
	bool_t 			ok;

	FOLLOW_label("preload_cache starts")

	if (idx >= egkey[key].maxindex) {
		FOLLOW_LULU("Wrong index", __LINE__, idx)	
		return FALSE;
	}
	
	/* find aged blocked in cache */
	pblock = point_block_to_replace();

	if (NULL == pblock)
		return FALSE;

	p = pblock->p_arr;

	if (Uncompressed) {

		index_t block = egtb_block_getnumber (key, side, idx);
		index_t n     = egtb_block_getsize   (key, idx);

		ok =	   egtb_file_beready (key)
				&& egtb_block_park   (key, block)
				&& egtb_block_read   (key, n, Buffer_packed)
				&& egtb_block_unpack (side, n, Buffer_packed, p);	

		FOLLOW_LULU("preload_cache", __LINE__, ok)

		assert (decoding_scheme() == 0 && GTB_scheme == 0);	

		if (ok) { Bytes_read = Bytes_read + (uint64_t) n; }

	} else {

        index_t block = 0;
		index_t n = 0;
		index_t z = 0;
		
		ok =	   egtb_file_beready (key);

		FOLLOW_LULU("preload_cache", __LINE__, ok)

		if (ok) {
			block = egtb_block_getnumber (key, side, idx);
			n     = egtb_block_getsize   (key, idx);
			z     = egtb_block_getsize_zipped (key, block);				
		}

		ok =	   ok
				&& egtb_block_park   (key, block);
		FOLLOW_LULU("preload_cache", __LINE__, ok)

		ok =	   ok	
				&& egtb_block_read   (key, z, Buffer_zipped);
		FOLLOW_LULU("preload_cache", __LINE__, ok)

		ok =	   ok		
				&& egtb_block_decode (key, z, Buffer_zipped, n, Buffer_packed);
		FOLLOW_LULU("preload_cache", __LINE__, ok)

		ok =	   ok		 
				&& egtb_block_unpack (side, n, Buffer_packed, p);	
		FOLLOW_LULU("preload_cache", __LINE__, ok)

		if (ok) { Bytes_read = Bytes_read + (uint64_t) z; }
	}

	if (ok) {

		index_t 		offset;
		index_t			remainder;
		split_index (dtm_cache.entries_per_block, idx, &offset, &remainder); 

		pblock->key    = key;
		pblock->side   = side;
		pblock->offset = offset;
	} else {
		/* make it unusable */
		pblock->key    = -1;
		pblock->side   = gtbNOSIDE;
		pblock->offset = gtbNOINDEX;
	}

	FOLLOW_LU("preload_cache?", ok)

	return ok;		
}

/****************************************************************************\
|
|
|						MEMORY ALLOCATION ZONE
|
|
\****************************************************************************/


mySHARED void
egtb_freemem (int i)
{
	if (egkey[i].status == STATUS_MALLOC) {
		assert (egkey[i].egt_w != NULL);
		assert (egkey[i].egt_b != NULL);	
		free (egkey[i].egt_w);
		free (egkey[i].egt_b);	
		egkey[i].egt_w = NULL;
		egkey[i].egt_b = NULL;
	}	
	egkey[i].status = STATUS_ABSENT;	
}

/***************************************************************************/

mySHARED bool_t
get_dtm (tbkey_t key, unsigned side, index_t idx, dtm_t *out, bool_t probe_hard_flag)
{
	bool_t found;

	if (probe_hard_flag) {
		dtm_cache.hard++;
	} else {
		dtm_cache.soft++;
	}

	if (get_dtm_from_cache (key, side, idx, out)) {
		dtm_cache.hits++;
		found = TRUE;
	} else if (probe_hard_flag) {
		dtm_cache.hardmisses++;
		found = preload_cache (key, side, idx) &&
				get_dtm_from_cache (key, side, idx, out);

		if (found) {
			Drive.hits++;			
		} else {
			Drive.miss++;					
		}
			

	} else {
		dtm_cache.softmisses++;
		found = FALSE;
	}
	return found;
}


static bool_t
get_dtm_from_cache (tbkey_t key, unsigned side, index_t idx, dtm_t *out)
{
	index_t 	offset;
	index_t		remainder;
	bool_t 		found;
	dtm_block_t	*p;

	if (!dtm_cache_is_on())
		return FALSE;

	split_index (dtm_cache.entries_per_block, idx, &offset, &remainder); 

	found = NULL != (p = dtm_cache_pointblock (key, side, idx));

	if (found) {
		*out = p->p_arr[remainder];
		movetotop(p);
	}

	FOLLOW_LU("get_dtm_from_cache ok?",found)

	return found;
}


static void
split_index (size_t entries_per_block, index_t i, index_t *o, index_t *r)
{
	index_t n;
	n  = i / (index_t) entries_per_block;
	*o = n * (index_t) entries_per_block;
	*r = i - *o;
	return;
}


static dtm_block_t *
point_block_to_replace (void)
{
	dtm_block_t *p, *t, *s;

	assert (0 == dtm_cache.n || dtm_cache.top != NULL);
	assert (0 == dtm_cache.n || dtm_cache.bot != NULL);
	assert (0 == dtm_cache.n || dtm_cache.bot->prev == NULL);
	assert (0 == dtm_cache.n || dtm_cache.top->next == NULL);

	/* no cache is being used */
	if (dtm_cache.max_blocks == 0)
		return NULL;

	if (dtm_cache.n > 0 && -1 == dtm_cache.top->key) {

		/* top entry is unusable, should be the one to replace*/
		p = dtm_cache.top;

	} else
	if (dtm_cache.n == 0) {
		
		assert (NULL != dtm_cache.entry);
		p = &dtm_cache.entry[dtm_cache.n++];
		dtm_cache.top = p;
		dtm_cache.bot = p;
	
		assert (NULL != p);
		p->prev = NULL;
		p->next = NULL;

	} else
	if (dtm_cache.n < dtm_cache.max_blocks) { /* add */

		assert (NULL != dtm_cache.entry);
		s = dtm_cache.top;
		p = &dtm_cache.entry[dtm_cache.n++];
		dtm_cache.top = p;
	
		assert (NULL != p && NULL != s);
		s->next = p;
		p->prev = s;
		p->next = NULL;

	} else if (1 < dtm_cache.max_blocks) { /* replace*/ 
		
		assert (NULL != dtm_cache.bot && NULL != dtm_cache.top);
		t = dtm_cache.bot;
		s = dtm_cache.top;

		dtm_cache.bot = t->next;
		dtm_cache.top = t;

		s->next = t;
		t->prev = s;

		assert (dtm_cache.top);
		dtm_cache.top->next = NULL;

		assert (dtm_cache.bot);
		dtm_cache.bot->prev = NULL;

		p = t;

	} else {
		
		assert (1 == dtm_cache.max_blocks);
		p =	dtm_cache.top;
		assert (p == dtm_cache.bot && p == dtm_cache.entry);
	}
	
	/* make the information content unusable, it will be replaced */
	p->key    = -1;
	p->side   = gtbNOSIDE;
	p->offset = gtbNOINDEX;

	return p;
}

static void
movetotop (dtm_block_t *t)
{
	dtm_block_t *s, *nx, *pv;

	assert (t != NULL);

	if (t->next == NULL) /* at the top already */
		return;

	/* detach */
	pv = t->prev;
	nx = t->next;

	if (pv == NULL)  /* at the bottom */
		dtm_cache.bot = nx;
	else 
		pv->next = nx;

	if (nx == NULL) /* at the top */
		dtm_cache.top = pv;
	else
		nx->prev = pv;

	/* relocate */
	s = dtm_cache.top;
	assert (s != NULL);
	if (s == NULL)
		dtm_cache.bot = t;	
	else
		s->next = t;

	t->next = NULL;
	t->prev = s;
	dtm_cache.top = t;

	return;
}

/****************************************************************************\
 *
 *
 *								INDEXING ZONE
 *
 *
 ****************************************************************************/

static void
init_indexing (int verbosity)
{
	index_t a,b,c,d,e,f;

	init_flipt ();

	a = init_kkidx     () ;	
	b = init_ppidx     () ;	
	c = init_aaidx     () ;
	d = init_aaa       () ;
	e = init_pp48_idx  () ;
	f = init_ppp48_idx () ;

	if (verbosity) {
		printf ("\nGTB supporting tables, Initialization\n");
		printf ("  Max    kk idx: %8d\n", (int)a );	
		printf ("  Max    pp idx: %8d\n", (int)b );	
		printf ("  Max    aa idx: %8d\n", (int)c );
		printf ("  Max   aaa idx: %8d\n", (int)d );
		printf ("  Max  pp48 idx: %8d\n", (int)e );
		printf ("  Max ppp48 idx: %8d\n", (int)f );
	}

	if (!reach_was_initialized())
		reach_init();

	/* testing used only in development stage */

#ifdef _MSC_VER
#pragma warning(disable:4127)
#endif

	if (0) {
		list_index ();
		printf ("\nTEST indexing functions\n");

		test_kaakb ();
		test_kaabk ();
		test_kaaak ();
		test_kabbk ();
	
		test_kapkb ();
		test_kabkp ();
	
		test_kppka ();

		test_kapkp ();
		test_kabpk();
		test_kaapk ();
	
		test_kappk ();	
		test_kaakp ();
		test_kppk ();
	 	test_kppkp ();
	 	test_kpppk ();
	}	

#ifdef _MSC_VER
#pragma warning(default:4127)
#endif

	return;
}


static index_t
init_kkidx (void)
/* modifies kkidx[][], wksq[], bksq[] */
{
	index_t idx;
	SQUARE x, y, i, j;
	
	/* default is noindex */
	for (x = 0; x < 64; x++) {
		for (y = 0; y < 64; y++) {
			IDX_set_empty(kkidx [x][y]);
		}
	}

	idx = 0;
	for (x = 0; x < 64; x++) {
		for (y = 0; y < 64; y++) {
		
			/* is x,y illegal? continue */
			if (possible_attack (x, y, wK) || x == y)
				continue;
		
			/* normalize */
			/*i <-- x; j <-- y */
			norm_kkindex (x, y, &i, &j);
		
			if (IDX_is_empty(kkidx [i][j])) { /* still empty */
				kkidx [i][j] = idx;
				kkidx [x][y] = idx;
				bksq [idx] = i;
				wksq [idx] = j;			
				idx++;
			}
		}
	}
	
	assert (idx == MAX_KKINDEX);

	return idx;
}


static index_t
init_aaidx (void)
/* modifies aabase[], aaidx[][] */
{
	index_t idx;
	SQUARE x, y;
	
	/* default is noindex */
	for (x = 0; x < 64; x++) {
		for (y = 0; y < 64; y++) {
			IDX_set_empty(aaidx [x][y]);
		}
	}

	for (idx = 0; idx < MAX_AAINDEX; idx++)
		aabase [idx] = 0;

	idx = 0;
	for (x = 0; x < 64; x++) {
		for (y = x + 1; y < 64; y++) {

			assert (idx == (int)((y - x) + x * (127-x)/2 - 1) );

			if (IDX_is_empty(aaidx [x][y])) { /* still empty */
				aaidx [x] [y] = idx; 
				aaidx [y] [x] = idx;
				aabase [idx] = (unsigned char) x;
				idx++;
			} else {
				assert (aaidx [x] [y] == idx);
				assert (aabase [idx] == x);
			}


		}
	}
	
	assert (idx == MAX_AAINDEX);

	return idx;
}


static index_t
init_ppidx (void)
/* modifies ppidx[][], pp_hi24[], pp_lo48[] */
{
	index_t i, j;
	index_t idx = 0;
	SQUARE a, b;

	/* default is noindex */
	for (i = 0; i < 24; i++) {
		for (j = 0; j < 48; j++) {
			IDX_set_empty(ppidx [i][j]);
		}
	}
		
	for (idx = 0; idx < MAX_PPINDEX; idx++) {
		IDX_set_empty(pp_hi24 [idx]);
		IDX_set_empty(pp_lo48 [idx]);		
	}		
		
	idx = 0;
	for (a = H7; a >= A2; a--) {

		if ((a & 07) < 4) /* square in the queen side */
			continue;

		for (b = a - 1; b >= A2; b--) {
			
			SQUARE anchor, loosen;
	
			pp_putanchorfirst (a, b, &anchor, &loosen);
			
			if ((anchor & 07) > 3) { /* square in the king side */
				anchor = flipWE(anchor);
				loosen = flipWE(loosen);				
			}
			
			i = wsq_to_pidx24 (anchor);
			j = wsq_to_pidx48 (loosen);
			
			if (IDX_is_empty(ppidx [i] [j])) {

                ppidx [i] [j] = idx;
                assert (idx < MAX_PPINDEX);
                pp_hi24 [idx] = i;
                assert (i < 24);
                pp_lo48 [idx] =	j;
                assert (j < 48);
				idx++;
			}
			
		}	
	}
	assert (idx == MAX_PPINDEX);
	return idx;
}

static void
init_flipt (void)
{
	unsigned int i, j;
	for (i = 0; i < 64; i++) {
		for (j = 0; j < 64; j++) {
			flipt [i] [j] = flip_type (i, j);
		}		
	}
}

/*--- NORMALIZE -------*/

static void
norm_kkindex (SQUARE x, SQUARE y, /*@out@*/ SQUARE *pi, /*@out@*/ SQUARE *pj)
{
	unsigned int rowx, rowy, colx, coly;
	
	assert (x < 64);
	assert (y < 64);
	
	if (getcol(x) > 3) { 
		x = flipWE (x); /* x = x ^ 07  */
		y = flipWE (y);		
	}
	if (getrow(x) > 3)  { 
		x = flipNS (x); /* x = x ^ 070  */
		y = flipNS (y);		
	}	
	rowx = getrow(x);
	colx = getcol(x);
	if ( rowx > colx ) {
		x = flipNW_SE (x); /* x = ((x&7)<<3) | (x>>3) */
		y = flipNW_SE (y);			
	}
	rowy = getrow(y);
	coly = getcol(y);	
	if ( rowx == colx && rowy > coly) {
		x = flipNW_SE (x);
		y = flipNW_SE (y);			
	}	
	
	*pi = x;
	*pj = y;
}

static unsigned int
flip_type (SQUARE x, SQUARE y)
{
	unsigned int rowx, rowy, colx, coly;
	unsigned int ret = 0;
	
	assert (x < 64);
	assert (y < 64);
	
	
	if (getcol(x) > 3) { 
		x = flipWE (x); /* x = x ^ 07  */
		y = flipWE (y);		
		ret |= 1;
	}
	if (getrow(x) > 3)  { 
		x = flipNS (x); /* x = x ^ 070  */
		y = flipNS (y);		
		ret |= 2;		
	}	
	rowx = getrow(x);
	colx = getcol(x);
	if ( rowx > colx ) {
		x = flipNW_SE (x); /* x = ((x&7)<<3) | (x>>3) */
		y = flipNW_SE (y);	
		ret |= 4;				
	}
	rowy = getrow(y);
	coly = getcol(y);	
	if ( rowx == colx && rowy > coly) {
		x = flipNW_SE (x);
		y = flipNW_SE (y);	
		ret |= 4;					
	}	
	return ret;
}


static void
pp_putanchorfirst (SQUARE a, SQUARE b, /*@out@*/ SQUARE *out_anchor, /*@out@*/ SQUARE *out_loosen)
{
	unsigned int anchor, loosen;
			
	unsigned int row_b, row_a;
	row_b = b & 070;
	row_a = a & 070;
			
	/* default */
	anchor = a;
	loosen = b;
	if (row_b > row_a) {
		anchor = b;
		loosen = a;
	} 
	else
	if (row_b == row_a) {
		unsigned int x, col, inv, hi_a, hi_b;
		x = a;
		col = x & 07;
		inv = col ^ 07;
		x = (1u<<col) | (1u<<inv);
		x &= (x-1);	
		hi_a = x;
		
		x = b;
		col = x & 07;
		inv = col ^ 07;
		x = (1u<<col) | (1u<<inv);
		x &= (x-1);	
		hi_b = x;			
				
		if (hi_b > hi_a) {
			anchor = b;
			loosen = a;					
		}

		if (hi_b < hi_a) {
			anchor = a;
			loosen = b;					
		}

		if (hi_b == hi_a) {
			if (a < b) {
				anchor = a;
				loosen = b;	
			} else {
				anchor = b;
				loosen = a;	
			}				
		}
	}

	*out_anchor = anchor;
	*out_loosen = loosen;
	return;
}


static index_t
wsq_to_pidx24 (SQUARE pawn)
{
	unsigned int idx24;
	SQUARE sq = pawn;

	/* input can be only queen side, pawn valid */
	assert (A2 <= pawn && pawn < A8);
	assert ((pawn & 07) < 4);

	sq ^= 070; /* flipNS*/
	sq -= 8;   /* down one row*/
	idx24 = (sq+(sq&3)) >> 1; 
	assert (idx24 < 24);
	return (index_t) idx24;
}

static index_t
wsq_to_pidx48 (SQUARE pawn)
{
	unsigned int idx48;
	SQUARE sq = pawn;

	/* input can be both queen or king side, pawn valid square  */
	assert (A2 <= pawn && pawn < A8);

	sq ^= 070; /* flipNS*/
	sq -= 8;   /* down one row*/
	idx48 = sq;
	assert (idx48 < 48);
	return (index_t)idx48;
}

static SQUARE
pidx24_to_wsq (index_t a)
{
	enum  {B11100  = 7u << 2};
	unsigned int x = (unsigned int) a; 	/* x is pslice */
	assert (a < 24);

	x += x & B11100; /* get upper part and double it */
	x += 8;          /* add extra row  */
	x ^= 070;        /* flip NS */
	return (SQUARE) x;
}

static SQUARE
pidx48_to_wsq (index_t a)
{
	unsigned int x;
	assert (a < 48);
	/* x is pslice */
	x = (unsigned int)a;
	x += 8;          /* add extra row  */
	x ^= 070;        /* flip NS */
	return x;
}


static void
kxk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {BLOCK_A = 64}; 

	index_t a = i / BLOCK_A;
	index_t b = i - a * BLOCK_A;
	
	pw[0] = wksq [a];
	pb[0] = bksq [a];
	pw[1] = (SQUARE) b;
	pw[2] = NOSQUARE;
	pb[1] = NOSQUARE;
	
	assert (kxk_pctoindex (pw, pb, &a) && a == i);

	return;
}

static bool_t 
kxk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {BLOCK_A = 64}; 
	SQUARE *p;
	SQUARE ws[32], bs[32];
	index_t ki;
	int i;
	
	unsigned int ft;
	
	ft = flip_type (inp_pb[0],inp_pw[0]);

	assert (ft < 8);


	for (i = 0; inp_pw[i] != NOSQUARE; i++) {
		ws[i] = inp_pw[i];
	}
	ws[i] = NOSQUARE;
	for (i = 0; inp_pb[i] != NOSQUARE; i++) {
		bs[i] = inp_pb[i];
	}
	bs[i] = NOSQUARE;

	if ((ft & 1) != 0) {
		for (p = ws; *p != NOSQUARE; p++) 
				*p = flipWE (*p);
		for (p = bs; *p != NOSQUARE; p++) 
				*p = flipWE (*p);
	}

	if ((ft & 2) != 0) {
		for (p = ws; *p != NOSQUARE; p++) 
				*p = flipNS (*p);
		for (p = bs; *p != NOSQUARE; p++) 
				*p = flipNS (*p);
	}

	if ((ft & 4) != 0) {
		for (p = ws; *p != NOSQUARE; p++) 
				*p = flipNW_SE (*p);
		for (p = bs; *p != NOSQUARE; p++) 
				*p = flipNW_SE (*p);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */

	if (IDX_is_empty(ki)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + (index_t) ws[1]; 
	return TRUE;
	
}


static void
kabk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	index_t a, b, c, r;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r;
	
	pw[0] = wksq [a];
	pb[0] = bksq [a];

	pw[1] = (SQUARE) b;
	pw[2] = (SQUARE) c;
	pw[3] = NOSQUARE;

	pb[1] = NOSQUARE;
	
	assert (kabk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kabk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	SQUARE *p;
	SQUARE ws[32], bs[32];
	index_t ki;
	int i;
	
	unsigned int ft;
	
	ft = flip_type (inp_pb[0],inp_pw[0]);

	assert (ft < 8);

	for (i = 0; inp_pw[i] != NOSQUARE; i++) {
		ws[i] = inp_pw[i];
	}
	ws[i] = NOSQUARE;
	for (i = 0; inp_pb[i] != NOSQUARE; i++) {
		bs[i] = inp_pb[i];
	}
	bs[i] = NOSQUARE;

	if ((ft & 1) != 0) {
		for (p = ws; *p != NOSQUARE; p++) 
				*p = flipWE (*p);
		for (p = bs; *p != NOSQUARE; p++) 
				*p = flipWE (*p);
	}

	if ((ft & 2) != 0) {
		for (p = ws; *p != NOSQUARE; p++) 
				*p = flipNS (*p);
		for (p = bs; *p != NOSQUARE; p++) 
				*p = flipNS (*p);
	}

	if ((ft & 4) != 0) {
		for (p = ws; *p != NOSQUARE; p++) 
				*p = flipNW_SE (*p);
		for (p = bs; *p != NOSQUARE; p++) 
				*p = flipNW_SE (*p);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */

	if (IDX_is_empty(ki)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + (index_t)ws[1] * BLOCK_B + (index_t)ws[2]; 
	return TRUE;
	
}


static void
kabkc_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;
	d  = r;
	
	pw[0] = wksq [a];
	pb[0] = bksq [a];

	pw[1] = (SQUARE) b;
	pw[2] = (SQUARE) c;
	pw[3] = NOSQUARE;

	pb[1] = (SQUARE) d;
	pb[2] = NOSQUARE;
	
	assert (kabkc_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kabkc_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {N_WHITE = 3, N_BLACK = 2};

	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki;
	int i;
	unsigned int ft;

	#if 0
		ft = flip_type (inp_pb[0], inp_pw[0]);
	#else
		ft = flipt [inp_pb[0]] [inp_pw[0]];
	#endif

	assert (ft < 8);

	for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i]; ws[N_WHITE] = NOSQUARE;
	for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i]; bs[N_BLACK] = NOSQUARE;	

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}


	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */

	if (IDX_is_empty(ki)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + (index_t)ws[1] * BLOCK_B + (index_t)ws[2] * BLOCK_C + (index_t)bs[1]; 
	return TRUE;
	
}

/* ABC/ ***/

extern void
kabck_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;
	d  = r;
	
	pw[0] = wksq [a];
	pb[0] = bksq [a];

	pw[1] = (SQUARE) b;
	pw[2] = (SQUARE) c;
	pw[3] = (SQUARE) d;
	pw[4] = NOSQUARE;

	pb[1] = NOSQUARE;
	
	assert (kabck_pctoindex (pw, pb, &a) && a == i);

	return;
}


extern bool_t 
kabck_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {N_WHITE = 4, N_BLACK = 1};
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 

	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki;
	int i;
	unsigned int ft;

	ft = flipt [inp_pb[0]] [inp_pw[0]];

	assert (ft < 8);

	for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i]; ws[N_WHITE] = NOSQUARE;
	for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i]; bs[N_BLACK] = NOSQUARE;	

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}


	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */

	if (IDX_is_empty(ki)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + (index_t)ws[1] * BLOCK_B + (index_t)ws[2] * BLOCK_C + (index_t)ws[3]; 
	return TRUE;
	
}


static void
kakb_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	index_t a, b, c, r;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r;
	
	pw[0] = wksq [a];
	pb[0] = bksq [a];

	pw[1] = (SQUARE) b;
	pw[2] = NOSQUARE;

	pb[1] = (SQUARE) c;
	pb[2] = NOSQUARE;
	
	assert (kakb_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kakb_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	SQUARE ws[32], bs[32];
	index_t ki;
	unsigned int ft;
	
	#if 0
		ft = flip_type (inp_pb[0], inp_pw[0]);
	#else
		ft = flipt [inp_pb[0]] [inp_pw[0]];
	#endif

	assert (ft < 8);

	ws[0] = inp_pw[0];
	ws[1] = inp_pw[1];
	ws[2] = NOSQUARE;
	
	bs[0] = inp_pb[0];
	bs[1] = inp_pb[1];
	bs[2] = NOSQUARE;

	if ((ft & 1) != 0) {
		ws[0] = flipWE (ws[0]);
		ws[1] = flipWE (ws[1]);
		bs[0] = flipWE (bs[0]);
		bs[1] = flipWE (bs[1]);
	}

	if ((ft & 2) != 0) {
		ws[0] = flipNS (ws[0]);
		ws[1] = flipNS (ws[1]);
		bs[0] = flipNS (bs[0]);
		bs[1] = flipNS (bs[1]);
	}

	if ((ft & 4) != 0) {
		ws[0] = flipNW_SE (ws[0]);
		ws[1] = flipNW_SE (ws[1]);
		bs[0] = flipNW_SE (bs[0]);
		bs[1] = flipNW_SE (bs[1]);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */

	if (IDX_is_empty(ki)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + (index_t)ws[1] * BLOCK_B + (index_t)bs[1]; 
	return TRUE;
	
}

/********************** KAAKB *************************************/

static bool_t 	test_kaakb (void);
static bool_t 	kaakb_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kaakb_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kaakb (void)
{

	enum  {MAXPC = 16+1};
	char 		str[] = "kaakb";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = d;
			pb[1] = e;
			pb[2] = NOSQUARE;
	
			if (kaakb_pctoindex (pw, pb, &i)) {
							kaakb_indextopc (i, px, py);		
							kaakb_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kaakb_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {
			BLOCK_B = 64,
			BLOCK_A = BLOCK_B * MAX_AAINDEX 
		}; 
	index_t a, b, c, r, x, y;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;

	b  = r / BLOCK_B;
	r -= b * BLOCK_B;

	c  = r;
	
	assert (i == (a * BLOCK_A + b * BLOCK_B + c));

	pw[0] = wksq [a];
	pb[0] = bksq [a];

	x = aabase [b];
	y = (b + 1) + x - (x * (127-x)/2);

	pw[1] = (SQUARE) x;
	pw[2] = (SQUARE) y;
	pw[3] = NOSQUARE;

	pb[1] = (SQUARE) c;
	pb[2] = NOSQUARE;
	
	assert (kaakb_pctoindex (pw, pb, &a) && a == i);

	return;
}

static bool_t 
kaakb_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out)
{
	enum  {N_WHITE = 3, N_BLACK = 2};
	enum  {
			BLOCK_B = 64,
			BLOCK_A = BLOCK_B * MAX_AAINDEX 
		}; 
	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki, ai;
	unsigned int ft;
	int i;

	ft = flipt [inp_pb[0]] [inp_pw[0]];

	assert (ft < 8);

	for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i]; ws[N_WHITE] = NOSQUARE;
	for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i]; bs[N_BLACK] = NOSQUARE;	

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */
	ai = aaidx [ws[1]] [ws[2]];

	if (IDX_is_empty(ki) || IDX_is_empty(ai)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + ai * BLOCK_B + (index_t)bs[1]; 
	return TRUE;
}

/****************** End KAAKB *************************************/

/********************** KAAB/K ************************************/

static bool_t 	test_kaabk (void);
static bool_t 	kaabk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kaabk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kaabk (void)
{

	enum  {MAXPC = 16+1};
	char 		str[] = "kaabk";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = d;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kaabk_pctoindex (pw, pb, &i)) {
							kaabk_indextopc (i, px, py);		
							kaabk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kaabk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {
			BLOCK_B = 64,
			BLOCK_A = BLOCK_B * MAX_AAINDEX 
		}; 
	index_t a, b, c, r, x, y;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;

	b  = r / BLOCK_B;
	r -= b * BLOCK_B;

	c  = r;
	
	assert (i == (a * BLOCK_A + b * BLOCK_B + c));

	pw[0] = wksq [a];
	pb[0] = bksq [a];

	x = aabase [b];
	y = (b + 1) + x - (x * (127-x)/2);

	pw[1] = (SQUARE) x;
	pw[2] = (SQUARE) y;
	pw[3] = (SQUARE) c;
	pw[4] = NOSQUARE;

	pb[1] = NOSQUARE;
	
	assert (kaabk_pctoindex (pw, pb, &a) && a == i);

	return;
}

static bool_t 
kaabk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out)
{
	enum  {N_WHITE = 4, N_BLACK = 1};
	enum  {
			BLOCK_B = 64,
			BLOCK_A = BLOCK_B * MAX_AAINDEX 
		}; 
	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki, ai;
	unsigned int ft;
	int i;

	ft = flipt [inp_pb[0]] [inp_pw[0]];

	assert (ft < 8);

	for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i]; ws[N_WHITE] = NOSQUARE;
	for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i]; bs[N_BLACK] = NOSQUARE;	

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */
	ai = aaidx [ws[1]] [ws[2]];

	if (IDX_is_empty(ki) || IDX_is_empty(ai)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + ai * BLOCK_B + (index_t)ws[3]; 
	return TRUE;
}

/****************** End KAAB/K *************************************/

/********************** KABB/K ************************************/

static bool_t 	test_kabbk (void);
static bool_t 	kabbk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kabbk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kabbk (void)
{

	enum  {MAXPC = 16+1};
	char 		str[] = "kabbk";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = d;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kabbk_pctoindex (pw, pb, &i)) {
							kabbk_indextopc (i, px, py);		
							kabbk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kabbk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {
			BLOCK_B = 64,
			BLOCK_A = BLOCK_B * MAX_AAINDEX 
		}; 
	index_t a, b, c, r, x, y;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;

	b  = r / BLOCK_B;
	r -= b * BLOCK_B;

	c  = r;
	
	assert (i == (a * BLOCK_A + b * BLOCK_B + c));

	pw[0] = wksq [a];
	pb[0] = bksq [a];

	x = aabase [b];
	y = (b + 1) + x - (x * (127-x)/2);

	pw[1] = (SQUARE) c;
	pw[2] = (SQUARE) x;
	pw[3] = (SQUARE) y;
	pw[4] = NOSQUARE;

	pb[1] = NOSQUARE;
	
	assert (kabbk_pctoindex (pw, pb, &a) && a == i);

	return;
}

static bool_t 
kabbk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, /*@out@*/ index_t *out)
{
	enum  {N_WHITE = 4, N_BLACK = 1};
	enum  {
			BLOCK_B = 64,
			BLOCK_A = BLOCK_B * MAX_AAINDEX 
		}; 
	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki, ai;
	unsigned int ft;
	int i;

	ft = flipt [inp_pb[0]] [inp_pw[0]];

	assert (ft < 8);

	for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i]; ws[N_WHITE] = NOSQUARE;
	for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i]; bs[N_BLACK] = NOSQUARE;	

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */
	ai = aaidx [ws[2]] [ws[3]];

	if (IDX_is_empty(ki) || IDX_is_empty(ai)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + ai * BLOCK_B + (index_t)ws[1]; 
	return TRUE;
}

/********************** End KABB/K *************************************/

/********************** init KAAA/K ************************************/

static index_t
aaa_getsubi (sq_t x, sq_t y, sq_t z);

static sq_t 		aaa_xyz [MAX_AAAINDEX] [3];
static index_t	 	aaa_base [64];

static index_t
init_aaa (void)
/* modifies aaa_base[], aaa_xyz[][] */
{
	index_t comb [64];
	index_t accum;
	index_t a;
	
	index_t idx;
	SQUARE x, y, z;

	/* getting aaa_base */	
	comb [0] = 0;	
	for (a = 1; a < 64; a++) {
		comb [a] = a * (a-1) / 2;	
	}
	
	accum = 0;
	aaa_base [0] = accum;		
	for (a = 0; a < (64-1); a++) {
		accum += comb[a];
		aaa_base [a+1] = accum;	
	}

	assert ((accum + comb[63]) == MAX_AAAINDEX);
	/* end getting aaa_base */


	/* initialize aaa_xyz [][] */
	for (idx = 0; idx < MAX_AAAINDEX; idx++) {
		IDX_set_empty (aaa_xyz[idx][0]);
		IDX_set_empty (aaa_xyz[idx][1]);				
		IDX_set_empty (aaa_xyz[idx][2]);
	}

	idx = 0;
	for (z = 0; z < 64; z++) {
		for (y = 0; y < z; y++) {
			for (x = 0; x < y; x++) {
			
				assert (idx == aaa_getsubi (x, y, z));
	
				aaa_xyz [idx] [0] = x;
				aaa_xyz [idx] [1] = y;				
				aaa_xyz [idx] [2] = z;
				
				idx++;
			}	
		}
	}
	
	assert (idx == MAX_AAAINDEX);

	return idx;
}


static index_t
aaa_getsubi (sq_t x, sq_t y, sq_t z)
/* uses aaa_base */
{
	index_t calc_idx, base;
	
	assert (x < 64 && y < 64 && z < 64);
	assert (x < y && y < z);

	base = aaa_base[z];
	calc_idx = (index_t)x + ((index_t)y - 1) * (index_t)y / 2 + base;

	return calc_idx;
}

/********************** KAAA/K ************************************/

static bool_t 	test_kaaak (void);
static bool_t 	kaaak_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kaaak_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kaaak (void)
{

	enum  {MAXPC = 16+1};
	char 		str[] = "kaaak";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = d;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kaaak_pctoindex (pw, pb, &i)) {
							kaaak_indextopc (i, px, py);		
							kaaak_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kaaak_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {
			BLOCK_A = MAX_AAAINDEX 
		};
	index_t a, b, r;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;

	b  = r;
	
	assert (i == (a * BLOCK_A + b));
	assert (b < BLOCK_A);

	pw[0] = wksq [a];
	pb[0] = bksq [a];

	pw[1] = aaa_xyz [b] [0];
	pw[2] = aaa_xyz [b] [1];
	pw[3] = aaa_xyz [b] [2];
	pw[4] = NOSQUARE;

	pb[1] = NOSQUARE;

	assert (kaaak_pctoindex (pw, pb, &a) && a == i);

	return;
}

static bool_t 
kaaak_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {N_WHITE = 4, N_BLACK = 1};
	enum  {
			BLOCK_A = MAX_AAAINDEX 
		}; 
	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki, ai;
	unsigned int ft;
	int i;

	ft = flipt [inp_pb[0]] [inp_pw[0]];

	assert (ft < 8);

	for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i]; ws[N_WHITE] = NOSQUARE;
	for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i]; bs[N_BLACK] = NOSQUARE;	

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}


	{ 
		SQUARE tmp;
		if (ws[2] < ws[1]) {
            tmp = ws[1];
            ws[1] = ws[2];
            ws[2] = tmp;
		}
		if (ws[3] < ws[2]) {
            tmp = ws[2];
            ws[2] = ws[3];
            ws[3] = tmp;
		}
		if (ws[2] < ws[1]) {
            tmp = ws[1];
            ws[1] = ws[2];
            ws[2] = tmp;
		}
	}
	
	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */

/*128 == (128 & (((ws[1]^ws[2])-1) | ((ws[1]^ws[3])-1) | ((ws[2]^ws[3])-1)) */
	
	if (ws[1] == ws[2] || ws[1] == ws[3] || ws[2] == ws[3]) {
		*out = NOINDEX;
		return FALSE;
	}
	
	ai = aaa_getsubi ( ws[1], ws[2], ws[3] );	
	
	if (IDX_is_empty(ki) || IDX_is_empty(ai)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + ai; 
	return TRUE;
}

/****************** End KAAB/K *************************************/

/**********************  KAP/KB ************************************/

static bool_t 	test_kapkb (void);
static bool_t 	kapkb_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void	kapkb_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kapkb (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kapkb";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (c <= H1 || c >= A8)
				continue;
			
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = d;
			pb[2] = NOSQUARE;
	
			if (kapkb_pctoindex (pw, pb, &i)) {
							kapkb_indextopc (i, px, py);		
							kapkb_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kapkb_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d * BLOCK_D + e; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64*64*64, BLOCK_B = 64*64*64, BLOCK_C = 64*64, BLOCK_D = 64}; 
	index_t a, b, c, d, e, r;
	index_t x;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r / BLOCK_D;
	r -= d * BLOCK_D;
	e  = r;
	
	/* x is pslice */
	x = a;
	x += x & B11100; /* get upper part and double it */
	x += 8;          /* add extra row  */
	x ^= 070;        /* flip NS */

	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;	
	pw[1] = (SQUARE) d;
	pw[2] = (SQUARE) x;
	pw[3] = NOSQUARE;
	pb[1] = (SQUARE) e;
	pb[2] = NOSQUARE;
	
	assert (kapkb_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kapkb_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64*64, BLOCK_B = 64*64*64, BLOCK_C = 64*64, BLOCK_D = 64}; 	
	index_t pslice;
	SQUARE sq;
	SQUARE pawn = pw[2];
	SQUARE wa   = pw[1];
	SQUARE wk   = pw[0];
	SQUARE bk   = pb[0];
	SQUARE ba   = pb[1];
	
	assert (A2 <= pawn && pawn < A8);

	if (  !(A2 <= pawn && pawn < A8)) {
		*out = NOINDEX;
		return FALSE;
	}	
	
	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
		ba   = flipWE (ba);		
	}

	sq = pawn;
	sq ^= 070; /* flipNS*/
	sq -= 8;   /* down one row*/
	pslice = (index_t) ((sq+(sq&3)) >> 1); 

	*out = pslice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk * BLOCK_C + (index_t)wa * BLOCK_D + (index_t)ba;

	return TRUE;
}
/********************** end KAP/KB ************************************/

/*************************  KAB/KP ************************************/

static bool_t 	test_kabkp (void);
static bool_t 	kabkp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void	kabkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kabkp (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kabkp";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (d <= H1 || d >= A8)
				continue;
			
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = d;
			pb[2] = NOSQUARE;
	
			if (kabkp_pctoindex (pw, pb, &i)) {
							kabkp_indextopc (i, px, py);		
							kabkp_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kabkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d * BLOCK_D + e; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64*64*64, BLOCK_B = 64*64*64, BLOCK_C = 64*64, BLOCK_D = 64}; 
	index_t a, b, c, d, e, r;
	index_t x;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r / BLOCK_D;
	r -= d * BLOCK_D;
	e  = r;
	
	/* x is pslice */
	x = a;
	x += x & B11100; /* get upper part and double it */
	x += 8;          /* add extra row  */
	/*x ^= 070;*/        /* do not flip NS */

	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;	
	pw[1] = (SQUARE) d;
	pw[2] = (SQUARE) e;
	pw[3] = NOSQUARE;
	pb[1] = (SQUARE) x;
	pb[2] = NOSQUARE;
	
	assert (kabkp_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kabkp_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64*64, BLOCK_B = 64*64*64, BLOCK_C = 64*64, BLOCK_D = 64}; 	
	index_t pslice;
	SQUARE sq;
	SQUARE pawn = pb[1];
	SQUARE wa   = pw[1];
	SQUARE wk   = pw[0];
	SQUARE bk   = pb[0];
	SQUARE wb   = pw[2];
	
	assert (A2 <= pawn && pawn < A8);

	if (  !(A2 <= pawn && pawn < A8)) {
		*out = NOINDEX;
		return FALSE;
	}	
	
	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
		wb   = flipWE (wb);		
	}

	sq = pawn;
	/*sq ^= 070;*/ /* do not flipNS*/
	sq -= 8;   /* down one row*/
	pslice = (index_t) ((sq+(sq&3)) >> 1); 

	*out = pslice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk * BLOCK_C + (index_t)wa * BLOCK_D + (index_t)wb;

	return TRUE;
}
/********************** end KAB/KP ************************************/


static void
kpk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	index_t a, b, c, r;
	index_t x;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r;
	
	/* x is pslice */
	x = a;
	x += x & B11100; /* get upper part and double it */
	x += 8;          /* add extra row  */
	x ^= 070;        /* flip NS */
	
	pw[1] = (SQUARE) x;
	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;

	pw[2] = NOSQUARE;
	pb[1] = NOSQUARE;
	
	assert (kpk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kpk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 	
	index_t pslice;
	SQUARE sq;
	SQUARE pawn = pw[1];
	SQUARE wk   = pw[0];
	SQUARE bk   = pb[0];

	#ifdef DEBUG
	if (  !(A2 <= pawn && pawn < A8)) {
		SQ_CONTENT wp[MAX_LISTSIZE], bp[MAX_LISTSIZE];
        bp [0] = wp[0] = KING;
        wp[1] = PAWN;
        wp[2] = bp[1] = NOPIECE;
		output_state (0, pw, pb, wp, bp);
	}	
	#endif

	assert (A2 <= pawn && pawn < A8);

	if (  !(A2 <= pawn && pawn < A8)) {
		*out = NOINDEX;
		return FALSE;
	}	
	
	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
	}

	sq = pawn;
	sq ^= 070; /* flipNS*/
	sq -= 8;   /* down one row*/
	pslice = (index_t) ((sq+(sq&3)) >> 1); 

	*out = pslice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk;

	return TRUE;
}


/**********************  KPP/K ************************************/

static bool_t 	test_kppk (void);
static bool_t 	kppk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kppk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kppk (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kppk";
	SQUARE 		a, b, c, d;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
			sq_t anchor1, anchor2, loosen1, loosen2;
			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;

			pp_putanchorfirst (b, c, &anchor1, &loosen1);
			pp_putanchorfirst (c, b, &anchor2, &loosen2);
			if (!(anchor1 == anchor2 && loosen1 == loosen2)) {
				printf ("Output depends on input in pp_outanchorfirst()\n input:%u, %u\n",(unsigned)b,(unsigned)c);
				fatal_error();
			} 
		}
	}


	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {


			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;

			
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = d;
			pb[1] = NOSQUARE;
	
			if (kppk_pctoindex (pw, pb, &i)) {
							kppk_indextopc (i, px, py);		
							kppk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}

static void
kppk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	index_t a, b, c, r;
	index_t m, n;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r;

	m = pp_hi24 [a];
	n = pp_lo48 [a];
	
	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;	
	pb[1] = NOSQUARE;	
	
	pw[1] = pidx24_to_wsq (m);
	pw[2] = pidx48_to_wsq (n);

	pw[3] = NOSQUARE;


	assert (A2 <= pw[1] && pw[1] < A8);
	assert (A2 <= pw[2] && pw[2] < A8);

#ifdef DEBUG
	if (!(kppk_pctoindex (pw, pb, &a) && a == i)) {
		pc_t wp[] = {KING, PAWN, PAWN, NOPIECE};
		pc_t bp[] = {KING, NOPIECE};		
		printf("Indexes not matching: input:%d, output:%d\n", i, a);
		print_pos (pw, pb, wp, bp);
	}
#endif

	assert (kppk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kppk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 	
	index_t pp_slice;
	SQUARE anchor, loosen;
	
	SQUARE wk     = pw[0];
	SQUARE pawn_a = pw[1];
	SQUARE pawn_b = pw[2];
	SQUARE bk     = pb[0];
	index_t i, j;

	#ifdef DEBUG
	if (!(A2 <= pawn_a && pawn_a < A8)) {
		printf ("\n\nsquare of pawn_a: %s\n", Square_str[pawn_a]);
		printf(" wk %s\n p1 %s\n p2 %s\n bk %s\n"
			, Square_str[wk]
			, Square_str[pawn_a]
			, Square_str[pawn_b]
			, Square_str[bk]
			);
	}
	#endif

	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);

	pp_putanchorfirst (pawn_a, pawn_b, &anchor, &loosen);

	if ((anchor & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		anchor = flipWE (anchor);
		loosen = flipWE (loosen);
		wk     = flipWE (wk);		
		bk     = flipWE (bk);		
	}
 
	i = wsq_to_pidx24 (anchor);
	j = wsq_to_pidx48 (loosen);

	pp_slice = ppidx [i] [j];

	if (IDX_is_empty(pp_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (pp_slice < MAX_PPINDEX );
	
	*out = pp_slice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk;

	return TRUE;
}
/****************** end  KPP/K ************************************/

static void
kakp_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;
	index_t x;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;
	
	/* x is pslice */
	x = a;
	x += x & B11100; /* get upper part and double it */
	x += 8;          /* add extra row  */
/*	x ^= 070;   */     /* flip NS */

	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;	
	pw[1] = (SQUARE) d;
	pb[1] = (SQUARE) x;
	pw[2] = NOSQUARE;
	pb[2] = NOSQUARE;
	
	assert (kakp_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kakp_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 	
	index_t pslice;
	SQUARE sq;
	SQUARE pawn = pb[1];
	SQUARE wa   = pw[1];
	SQUARE wk   = pw[0];
	SQUARE bk   = pb[0];

	assert (A2 <= pawn && pawn < A8);

	if (  !(A2 <= pawn && pawn < A8)) {
		*out = NOINDEX;
		return FALSE;
	}	
	
	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
	}

	sq = pawn;
	/*sq ^= 070;*/ /* flipNS*/
	sq -= 8;   /* down one row*/
	pslice = (index_t) ((sq+(sq&3)) >> 1); 

	*out = pslice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk * BLOCK_C + (index_t)wa;

	return TRUE;
}


static void
kapk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;
	index_t x;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;
	
	/* x is pslice */
	x = a;
	x += x & B11100; /* get upper part and double it */
	x += 8;          /* add extra row  */
	x ^= 070;        /* flip NS */

	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;	
	pw[1] = (SQUARE) d;
	pw[2] = (SQUARE) x;
	pw[3] = NOSQUARE;
	pb[1] = NOSQUARE;
	
	assert (kapk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kapk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 	
	index_t pslice;
	SQUARE sq;
	SQUARE pawn = pw[2];
	SQUARE wa   = pw[1];
	SQUARE wk   = pw[0];
	SQUARE bk   = pb[0];

	assert (A2 <= pawn && pawn < A8);

	if (  !(A2 <= pawn && pawn < A8)) {
		*out = NOINDEX;
		return FALSE;
	}	
	
	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
	}

	sq = pawn;
	sq ^= 070; /* flipNS*/
	sq -= 8;   /* down one row*/
	pslice = (index_t) ((sq+(sq&3)) >> 1); 

	*out = pslice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk * BLOCK_C + (index_t)wa;

	return TRUE;
}


static void
kaak_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	enum  {BLOCK_A = MAX_AAINDEX}; 
	index_t a, b, r, x, y;

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r;
	
	assert (i == (a * BLOCK_A + b));

	pw[0] = wksq [a];
	pb[0] = bksq [a];

	x = aabase [b];
	y = (b + 1) + x - (x * (127-x)/2);

	pw[1] = (SQUARE) x;
	pw[2] = (SQUARE) y;
	pw[3] = NOSQUARE;

	pb[1] = NOSQUARE;
	
	assert (kaak_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kaak_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out)
{
	enum  {N_WHITE = 3, N_BLACK = 1};
	enum  {BLOCK_A = MAX_AAINDEX}; 
	SQUARE ws[MAX_LISTSIZE], bs[MAX_LISTSIZE];
	index_t ki, ai;
	unsigned int ft;
	SQUARE i;

	ft = flipt [inp_pb[0]] [inp_pw[0]];

	assert (ft < 8);

    for (i = 0; i < N_WHITE; i++) ws[i] = inp_pw[i];
    ws[N_WHITE] = NOSQUARE;
    for (i = 0; i < N_BLACK; i++) bs[i] = inp_pb[i];
    bs[N_BLACK] = NOSQUARE;

	if ((ft & WE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipWE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipWE (bs[i]);
	}

	if ((ft & NS_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNS (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNS (bs[i]);
	}

	if ((ft & NW_SE_FLAG) != 0) {
		for (i = 0; i < N_WHITE; i++) ws[i] = flipNW_SE (ws[i]);
		for (i = 0; i < N_BLACK; i++) bs[i] = flipNW_SE (bs[i]);
	}

	ki = kkidx [bs[0]] [ws[0]]; /* kkidx [black king] [white king] */
	ai = (index_t) aaidx [ws[1]] [ws[2]];

	if (IDX_is_empty(ki) || IDX_is_empty(ai)) {
		*out = NOINDEX;
		return FALSE;
	}	
	*out = ki * BLOCK_A + ai; 
	return TRUE;
}

/**********************  KPP/KA ************************************/

static bool_t 	test_kppka (void);
static bool_t 	kppka_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kppka_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kppka (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kppka";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;

			
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = d;
			pb[2] = NOSQUARE;
	
			if (kppka_pctoindex (pw, pb, &i)) {
							kppka_indextopc (i, px, py);		
							kppka_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kppka_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/

	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;
	index_t m, n;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;
	
	m = pp_hi24 [a];
	n = pp_lo48 [a];
	
	pw[0] = (SQUARE) b;
	pw[1] = pidx24_to_wsq (m);
	pw[2] = pidx48_to_wsq (n);
	pw[3] = NOSQUARE;

	pb[0] = (SQUARE) c;	
	pb[1] = (SQUARE) d;
	pb[2] = NOSQUARE;	


	assert (A2 <= pw[1] && pw[1] < A8);
	assert (A2 <= pw[2] && pw[2] < A8);
	assert (kppka_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kppka_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t pp_slice;
	index_t i, j;

	SQUARE anchor, loosen;
	
	SQUARE wk     = pw[0];
	SQUARE pawn_a = pw[1];
	SQUARE pawn_b = pw[2];
	SQUARE bk     = pb[0];
	SQUARE ba	  = pb[1];	


	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);

	pp_putanchorfirst (pawn_a, pawn_b, &anchor, &loosen);

	if ((anchor & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		anchor = flipWE (anchor);
		loosen = flipWE (loosen);
		wk     = flipWE (wk);		
		bk     = flipWE (bk);		
		ba	   = flipWE (ba);	
	}
 
	i = wsq_to_pidx24 (anchor);
	j = wsq_to_pidx48 (loosen);

	pp_slice = ppidx [i] [j];

	if (IDX_is_empty(pp_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (pp_slice < MAX_PPINDEX );
	
	*out = pp_slice * (index_t)BLOCK_A + (index_t)wk * (index_t)BLOCK_B  + (index_t)bk * (index_t)BLOCK_C + (index_t)ba;

	return TRUE;
}

/********************** end KPP/KA ************************************/

/**********************  KAPP/K ************************************/

static bool_t 	test_kappk (void);
static bool_t 	kappk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kappk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kappk (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kappk";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;

			
			pw[0] = a;
			pw[1] = d;	
			pw[2] = b;
			pw[3] = c;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kappk_pctoindex (pw, pb, &i)) {
							kappk_indextopc (i, px, py);		
							kappk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kappk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/

	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;
	index_t m, n;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;
	
	m = pp_hi24 [a];
	n = pp_lo48 [a];
	
	pw[0] = (SQUARE) b;
	pw[1] = (SQUARE) d;
	pw[2] = pidx24_to_wsq (m);
	pw[3] = pidx48_to_wsq (n);
	pw[4] = NOSQUARE;

	pb[0] = (SQUARE) c;	
	pb[1] = NOSQUARE;	


	assert (A2 <= pw[3] && pw[3] < A8);
	assert (A2 <= pw[2] && pw[2] < A8);
	assert (kappk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kappk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t pp_slice;
	SQUARE anchor, loosen;
	
	SQUARE wk     = pw[0];
	SQUARE wa	  = pw[1];	
	SQUARE pawn_a = pw[2];
	SQUARE pawn_b = pw[3];
	SQUARE bk     = pb[0];

	index_t i, j;

	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);

	pp_putanchorfirst (pawn_a, pawn_b, &anchor, &loosen);

	if ((anchor & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		anchor = flipWE (anchor);
		loosen = flipWE (loosen);
		wk     = flipWE (wk);		
		bk     = flipWE (bk);		
		wa	   = flipWE (wa);	
	}
 
	i = wsq_to_pidx24 (anchor);
	j = wsq_to_pidx48 (loosen);

	pp_slice = ppidx [i] [j];

	if (IDX_is_empty(pp_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (pp_slice < MAX_PPINDEX );
	
	*out = pp_slice * (index_t)BLOCK_A + (index_t)wk * (index_t)BLOCK_B  + (index_t)bk * (index_t)BLOCK_C + (index_t)wa;

	return TRUE;
}

/********************** end KAPP/K ************************************/

/**********************  KAPP/K ************************************/

static bool_t 	test_kapkp (void);
static bool_t 	kapkp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kapkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kapkp (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kapkp";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;

			
			pw[0] = a;
			pw[1] = d;	
			pw[2] = b;
			pw[3] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = c;
			pb[2] = NOSQUARE;
	
			if (kapkp_pctoindex (pw, pb, &i)) {
							kapkp_indextopc (i, px, py);		
							kapkp_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static bool_t 
kapkp_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 	
	index_t pp_slice;
	SQUARE anchor, loosen;
	
	SQUARE wk     = pw[0];
	SQUARE wa	  = pw[1];
	SQUARE pawn_a = pw[2];
	SQUARE bk     = pb[0];
	SQUARE pawn_b = pb[1];
	index_t m, n;

	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);
	assert (pw[3] == NOSQUARE && pb[2] == NOSQUARE);

	anchor = pawn_a;
	loosen = pawn_b;

	if ((anchor & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		anchor = flipWE (anchor);
		loosen = flipWE (loosen);
		wk     = flipWE (wk);		
		bk     = flipWE (bk);		
		wa	   = flipWE (wa);		
	}
 
	m = wsq_to_pidx24 (anchor);
	n = (index_t)loosen - 8;

	pp_slice = m * 48 + n; 

	if (IDX_is_empty(pp_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (pp_slice < (64*MAX_PpINDEX) );
	
	*out = pp_slice * (index_t)BLOCK_A + (index_t)wk * (index_t)BLOCK_B  + (index_t)bk * (index_t)BLOCK_C + (index_t)wa;

	return TRUE;
}

static void
kapkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/
	enum  {BLOCK_A = 64*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	enum  {block_m = 48};
	index_t a, b, c, d, r;
	index_t m, n;
	SQUARE sq_m, sq_n;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;
	
	/* unpack a, which is pslice, into m and n */
	r = a;
	m  = r / block_m;
	r -= m * block_m;
	n  = r ;

	sq_m = pidx24_to_wsq (m);
	sq_n = (SQUARE) n + 8;
	
	pw[0] = (SQUARE) b;
	pb[0] = (SQUARE) c;	
	pw[1] = (SQUARE) d;
	pw[2] = sq_m;
	pb[1] = sq_n;
	pw[3] = NOSQUARE;
	pb[2] = NOSQUARE;	
	
	assert (A2 <= sq_m && sq_m < A8);
	assert (A2 <= sq_n && sq_n < A8);
	assert (kapkp_pctoindex (pw, pb, &a) && a == i);

	return;
}

/********************** end KAP/KP ************************************/

/**********************  KABP/K ************************************/

static bool_t 	test_kabpk (void);
static bool_t 	kabpk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kabpk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kabpk (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kabpk";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (d <= H1 || d >= A8)
				continue;
		
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = d;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kabpk_pctoindex (pw, pb, &i)) {
							kabpk_indextopc (i, px, py);		
							kabpk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kabpk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d * BLOCK_D + e; 
	*----------------------------------------------------------*/
	enum  {BLOCK_A = 64*64*64*64, BLOCK_B = 64*64*64, BLOCK_C = 64*64, BLOCK_D = 64}; 
	index_t a, b, c, d, e, r;
	SQUARE x;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r / BLOCK_D;
	r -= d * BLOCK_D;
	e  = r;
	
	x = pidx24_to_wsq(a);

	pw[0] = (SQUARE) b;
	pw[1] = (SQUARE) d;
	pw[2] = (SQUARE) e;
	pw[3] = x;
	pw[4] = NOSQUARE;

	pb[0] = (SQUARE) c;	
	pb[1] = NOSQUARE;
	
	assert (kabpk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kabpk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64*64*64, BLOCK_B = 64*64*64, BLOCK_C = 64*64, BLOCK_D = 64}; 	
	index_t pslice;

	SQUARE wk   = pw[0];
	SQUARE wa   = pw[1];
	SQUARE wb   = pw[2];
	SQUARE pawn = pw[3];
	SQUARE bk   = pb[0];

	assert (A2 <= pawn && pawn < A8);
	
	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
		wb   = flipWE (wb);		
	}

	pslice = wsq_to_pidx24 (pawn);

	*out = pslice * (index_t)BLOCK_A + (index_t)wk * (index_t)BLOCK_B  + (index_t)bk * (index_t)BLOCK_C + (index_t)wa * (index_t)BLOCK_D + (index_t)wb;

	return TRUE;
}

/********************** end KABP/K ************************************/

/**********************  KAAP/K ************************************/

static bool_t 	test_kaapk (void);
static bool_t 	kaapk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kaapk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kaapk (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kaapk";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (d <= H1 || d >= A8)
				continue;
		
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = d;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kaapk_pctoindex (pw, pb, &i)) {
							kaapk_indextopc (i, px, py);		
							kaapk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kaapk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/
	enum 	{BLOCK_C = MAX_AAINDEX
			,BLOCK_B = 64*BLOCK_C
			,BLOCK_A = 64*BLOCK_B
	}; 	
	index_t a, b, c, d, r;
	index_t x, y, z;
	
	assert (i >= 0);

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;

	z = (index_t) pidx24_to_wsq(a);
	
	/* split d into x, y*/
	x = aabase [d];
	y = (d + 1) + x - (x * (127-x)/2);

	assert (aaidx[x][y] == aaidx[y][x]);
	assert (aaidx[x][y] == d);


	pw[0] = (SQUARE) b;
	pw[1] = (SQUARE) x;
	pw[2] = (SQUARE) y;
	pw[3] = (SQUARE) z;
	pw[4] = NOSQUARE;
	
	pb[0] = (SQUARE) c;	
	pb[1] = NOSQUARE;
	
	assert (kaapk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kaapk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum 	{BLOCK_C = MAX_AAINDEX
			,BLOCK_B = 64*BLOCK_C
			,BLOCK_A = 64*BLOCK_B
	}; 	
	index_t aa_combo, pslice;

	SQUARE wk   = pw[0];
	SQUARE wa   = pw[1];
	SQUARE wa2  = pw[2];
	SQUARE pawn = pw[3];
	SQUARE bk   = pb[0];

	assert (A2 <= pawn && pawn < A8);

	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
		wa2  = flipWE (wa2);
	}

	pslice = wsq_to_pidx24 (pawn);

	aa_combo = (index_t) aaidx [wa] [wa2];

	if (IDX_is_empty(aa_combo)) {
		*out = NOINDEX;
		return FALSE;
	}	

	*out = pslice * (index_t)BLOCK_A + (index_t)wk * (index_t)BLOCK_B  + (index_t)bk * (index_t)BLOCK_C + aa_combo;

	assert (*out >= 0);

	return TRUE;
}

/********************** end KAAP/K ************************************/

/**********************  KAA/KP ************************************/

static bool_t 	test_kaakp (void);
static bool_t 	kaakp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kaakp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static bool_t
test_kaakp (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kaakp";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (d <= H1 || d >= A8)
				continue;
		
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = d;
			pb[2] = NOSQUARE;
	
			if (kaakp_pctoindex (pw, pb, &i)) {
							kaakp_indextopc (i, px, py);		
							kaakp_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kaakp_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/
	enum 	{BLOCK_C = MAX_AAINDEX
			,BLOCK_B = 64*BLOCK_C
			,BLOCK_A = 64*BLOCK_B
	}; 	
	index_t a, b, c, d, r;
	index_t x, y, z;
	SQUARE zq;	

	assert (i >= 0);

	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;

	zq = pidx24_to_wsq(a); 
	z  = (index_t)flipNS(zq);

	
	/* split d into x, y*/
	x = aabase [d];
	y = (d + 1) + x - (x * (127-x)/2);

	assert (aaidx[x][y] == aaidx[y][x]);
	assert (aaidx[x][y] == d);


	pw[0] = (SQUARE)b;
	pw[1] = (SQUARE)x;
	pw[2] = (SQUARE)y;
	pw[3] = NOSQUARE;
	
	pb[0] = (SQUARE)c;	
	pb[1] = (SQUARE)z;
	pb[2] = NOSQUARE;
	
	assert (kaakp_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kaakp_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum 	{BLOCK_C = MAX_AAINDEX
			,BLOCK_B = 64*BLOCK_C
			,BLOCK_A = 64*BLOCK_B
	}; 	
	index_t aa_combo, pslice;

	SQUARE wk   = pw[0];
	SQUARE wa   = pw[1];
	SQUARE wa2  = pw[2];
	SQUARE bk   = pb[0];
	SQUARE pawn = pb[1];

	assert (A2 <= pawn && pawn < A8);

	if ((pawn & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		pawn = flipWE (pawn);
		wk   = flipWE (wk);		
		bk   = flipWE (bk);		
		wa   = flipWE (wa);
		wa2  = flipWE (wa2);
	}

	pawn = flipNS(pawn);
	pslice = wsq_to_pidx24 (pawn);

	aa_combo = (index_t)aaidx [wa] [wa2];

	if (IDX_is_empty(aa_combo)) {
		*out = NOINDEX;
		return FALSE;
	}	

	*out = pslice * (index_t)BLOCK_A + (index_t)wk * (index_t)BLOCK_B  + (index_t)bk * (index_t)BLOCK_C + aa_combo;

	assert (*out >= 0);

	return TRUE;
}

/********************** end KAA/KP ************************************/

/**********************  KPP/KP ************************************/
/*
index_t 	pp48_idx[48][48];
sq_t		pp48_sq_x[MAX_PP48_INDEX];
sq_t		pp48_sq_y[MAX_PP48_INDEX]; 
*/
static bool_t 	test_kppkp (void);
static bool_t 	kppkp_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kppkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static sq_t map24_b   (sq_t s);
static sq_t unmap24_b (index_t i);

static index_t
init_pp48_idx (void)
/* modifies pp48_idx[][], pp48_sq_x[], pp48_sq_y[] */
{
	enum  {MAX_I = 48, MAX_J = 48};
	SQUARE i, j;
	index_t idx = 0;
	SQUARE a, b;

	/* default is noindex */
	for (i = 0; i < MAX_I; i++) {
		for (j = 0; j < MAX_J; j++) {
			IDX_set_empty (pp48_idx [i][j]);
		}
	}
		
	for (idx = 0; idx < MAX_PP48_INDEX; idx++) {
		pp48_sq_x [idx] = NOSQUARE;	
		pp48_sq_y [idx] = NOSQUARE;			
	}		
		
	idx = 0;
	for (a = H7; a >= A2; a--) {

		for (b = a - 1; b >= A2; b--) {

			i = flipWE( flipNS (a) ) - 8;
			j = flipWE( flipNS (b) ) - 8;
			
			if (IDX_is_empty(pp48_idx [i] [j])) {

				pp48_idx  [i][j]= idx; 	assert (idx < MAX_PP48_INDEX);
				pp48_idx  [j][i]= idx;
				pp48_sq_x [idx] = i; 	assert (i < MAX_I);
				pp48_sq_y [idx] = j; 	assert (j < MAX_J);		
				idx++;
			}
		}	
	}
	assert (idx == MAX_PP48_INDEX);
	return idx;
}



static bool_t
test_kppkp (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kppkp";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;
			if (d <= H1 || d >= A8)
				continue;
			
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = d;
			pb[2] = NOSQUARE;
	
			if (kppkp_pctoindex (pw, pb, &i)) {
							kppkp_indextopc (i, px, py);		
							kppkp_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kppkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/

	enum  {BLOCK_A = MAX_PP48_INDEX*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t a, b, c, d, r;
	SQUARE m, n;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r / BLOCK_C;
	r -= c * BLOCK_C;	
	d  = r;
	
	m = pp48_sq_x [b];
	n = pp48_sq_y [b];
	
	pw[0] = (SQUARE)c;
	pw[1] = flipWE(flipNS(m+8));
	pw[2] = flipWE(flipNS(n+8));
	pw[3] = NOSQUARE;

	pb[0] = (SQUARE)d;	
	pb[1] = (SQUARE)unmap24_b (a);
	pb[2] = NOSQUARE;	


	assert (A2 <= pw[1] && pw[1] < A8);
	assert (A2 <= pw[2] && pw[2] < A8);
	assert (A2 <= pb[1] && pb[1] < A8);
	assert (kppkp_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kppkp_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = MAX_PP48_INDEX*64*64, BLOCK_B = 64*64, BLOCK_C = 64}; 
	index_t pp48_slice;	
	
	SQUARE wk     = pw[0];
	SQUARE pawn_a = pw[1];
	SQUARE pawn_b = pw[2];
	SQUARE bk     = pb[0];
	SQUARE pawn_c = pb[1];	
	SQUARE i, j, k;

	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);
	assert (A2 <= pawn_c && pawn_c < A8);

	if ((pawn_c & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		wk     = flipWE (wk);		
		pawn_a = flipWE (pawn_a);
		pawn_b = flipWE (pawn_b);
		bk     = flipWE (bk);		
		pawn_c = flipWE (pawn_c);	
	}
 
	i = flipWE( flipNS (pawn_a) ) - 8;
	j = flipWE( flipNS (pawn_b) ) - 8;
	k = map24_b (pawn_c); /* black pawn, so low indexes mean more advanced 0 == A2 */

	pp48_slice = pp48_idx [i] [j];

	if (IDX_is_empty(pp48_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (pp48_slice < MAX_PP48_INDEX );
	
	*out = (index_t)k * (index_t)BLOCK_A + pp48_slice * (index_t)BLOCK_B + (index_t)wk * (index_t)BLOCK_C  + (index_t)bk;

	return TRUE;
}

static sq_t
map24_b (sq_t s)
{
	s -= 8;
	return ((s&3)+s)>>1;
}

static sq_t
unmap24_b (index_t i)
{
	return (sq_t) ((i&(4+8+16)) + i + 8);
}

/********************** end KPP/KP ************************************/

/**********************  KPPP/K ************************************/

static const sq_t itosq[48] = {
	H7,G7,F7,E7,
	H6,G6,F6,E6,
	H5,G5,F5,E5,
	H4,G4,F4,E4,
	H3,G3,F3,E3,
	H2,G2,F2,E2,
	D7,C7,B7,A7,
	D6,C6,B6,A6,
	D5,C5,B5,A5,
	D4,C4,B4,A4,
	D3,C3,B3,A3,
	D2,C2,B2,A2
};

static bool_t 	test_kpppk (void);
static bool_t 	kpppk_pctoindex (const SQUARE *inp_pw, const SQUARE *inp_pb, index_t *out);
static void		kpppk_indextopc (index_t i, SQUARE *pw, SQUARE *pb);

static index_t
init_ppp48_idx (void)
/* modifies ppp48_idx[][], ppp48_sq_x[], ppp48_sq_y[], ppp48_sq_z[] */
{
	enum  {MAX_I = 48, MAX_J = 48, MAX_K = 48};
	SQUARE i, j, k;
	index_t idx = 0;
	SQUARE a, b, c;
	int x, y, z;

	/* default is noindex */
	for (i = 0; i < MAX_I; i++) {
		for (j = 0; j < MAX_J; j++) {
			for (k = 0; k < MAX_K; k++) {
				IDX_set_empty(ppp48_idx [i][j][k]);
			}
		}
	}
		
	for (idx = 0; idx < MAX_PPP48_INDEX; idx++) {
		ppp48_sq_x [idx] = (uint8_t)NOSQUARE;	
		ppp48_sq_y [idx] = (uint8_t)NOSQUARE;	
		ppp48_sq_z [idx] = (uint8_t)NOSQUARE;		
	}		

	idx = 0;
	for (x = 0; x < 48; x++) {
		for (y = x+1; y < 48; y++) {
			for (z = y+1; z < 48; z++) {

				a = itosq [x];
				b = itosq [y];
				c = itosq [z];
		
				if (!in_queenside(b) || !in_queenside(c))			
						continue;

				i = a - 8;
				j = b - 8;
				k = c - 8;
				
				if (IDX_is_empty(ppp48_idx [i] [j] [k])) {

					ppp48_idx  [i][j][k] = idx; 	
					ppp48_idx  [i][k][j] = idx;
					ppp48_idx  [j][i][k] = idx;
					ppp48_idx  [j][k][i] = idx;
					ppp48_idx  [k][i][j] = idx;
					ppp48_idx  [k][j][i] = idx;
					ppp48_sq_x [idx] = (uint8_t) i; 	assert (i < MAX_I);
					ppp48_sq_y [idx] = (uint8_t) j; 	assert (j < MAX_J);		
					ppp48_sq_z [idx] = (uint8_t) k; 	assert (k < MAX_K);	
					idx++;
				}
			}
		}	
	}

/*	assert (idx == MAX_PPP48_INDEX);*/
	return idx;
}

static bool_t
test_kpppk (void)
{

	enum 		{MAXPC = 16+1};
	char 		str[] = "kpppk";
	SQUARE 		a, b, c, d, e;
	SQUARE 		pw[MAXPC], pb[MAXPC];
	SQUARE 		px[MAXPC], py[MAXPC];	

	index_t		i, j;
	bool_t 		err = FALSE;

	printf ("%8s ", str);

	for (a = 0; a < 64; a++) {
		for (b = 0; b < 64; b++) {
		for (c = 0; c < 64; c++) {
		for (d = 0; d < 64; d++) {
		for (e = 0; e < 64; e++) {

			if (c <= H1 || c >= A8)
				continue;
			if (b <= H1 || b >= A8)
				continue;
			if (d <= H1 || d >= A8)
				continue;
			
			pw[0] = a;
			pw[1] = b;	
			pw[2] = c;
			pw[3] = d;
			pw[4] = NOSQUARE;
		
			pb[0] = e;
			pb[1] = NOSQUARE;
	
			if (kpppk_pctoindex (pw, pb, &i)) {
							kpppk_indextopc (i, px, py);		
							kpppk_pctoindex (px, py, &j);
							if (i != j) {
								err = TRUE;
							}
							assert (i == j);	
			}
	
		}
		}
		}
		}

        if ((a&1)==0) {
            printf(".");
            fflush(stdout);
        }
	}

	if (err)		
		printf ("> %s NOT passed\n", str);	
	else
		printf ("> %s PASSED\n", str);	
	return !err;
}


static void
kpppk_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c * BLOCK_C + d; 
	*----------------------------------------------------------*/

	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	index_t a, b, c, r;
	SQUARE m, n, o;
	
	r  = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r;
	
	m = ppp48_sq_x [a];
	n = ppp48_sq_y [a];
	o = ppp48_sq_z [a];

	
	pw[0] = (SQUARE)b;
	pw[1] = m + 8;
	pw[2] = n + 8;
	pw[3] = o + 8;
	pw[4] = NOSQUARE;

	pb[0] = (SQUARE)c;	
	pb[1] = NOSQUARE;	


	assert (A2 <= pw[1] && pw[1] < A8);
	assert (A2 <= pw[2] && pw[2] < A8);
	assert (A2 <= pw[3] && pw[3] < A8);
	assert (kpppk_pctoindex (pw, pb, &a) && a == i);

	return;
}


static bool_t 
kpppk_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	index_t ppp48_slice;	
	
	SQUARE wk     = pw[0];
	SQUARE pawn_a = pw[1];
	SQUARE pawn_b = pw[2];
	SQUARE pawn_c = pw[3];	

	SQUARE bk     = pb[0];

	SQUARE i, j, k;

	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);
	assert (A2 <= pawn_c && pawn_c < A8);

	i = pawn_a - 8;
	j = pawn_b - 8;
	k = pawn_c - 8;

	ppp48_slice = ppp48_idx [i] [j] [k];

	if (IDX_is_empty(ppp48_slice)) { 
		wk     = flipWE (wk);		
		pawn_a = flipWE (pawn_a);
		pawn_b = flipWE (pawn_b);
		pawn_c = flipWE (pawn_c);
		bk     = flipWE (bk);		
	}

	i = pawn_a - 8;
	j = pawn_b - 8;
	k = pawn_c - 8;

	ppp48_slice = ppp48_idx [i] [j] [k];
 
	if (IDX_is_empty(ppp48_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (ppp48_slice < MAX_PPP48_INDEX );
	
	*out = (index_t)ppp48_slice * BLOCK_A + (index_t)wk * BLOCK_B  + (index_t)bk;

	return TRUE;
}


/********************** end KPPP/K ************************************/


static bool_t 
kpkp_pctoindex (const SQUARE *pw, const SQUARE *pb, index_t *out)
{
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 	
	SQUARE pp_slice;	
	SQUARE anchor, loosen;
	
	SQUARE wk     = pw[0];
	SQUARE bk     = pb[0];
	SQUARE pawn_a = pw[1];
	SQUARE pawn_b = pb[1];

	SQUARE m, n;

	#ifdef DEBUG
	if (!(A2 <= pawn_a && pawn_a < A8)) {
		printf ("\n\nsquare of pawn_a: %s\n", Square_str[pawn_a]);
		printf(" wk %s\n p1 %s\n p2 %s\n bk %s\n"
			, Square_str[wk]
			, Square_str[pawn_a]
			, Square_str[pawn_b]
			, Square_str[bk]
			);
	}
	#endif

	assert (A2 <= pawn_a && pawn_a < A8);
	assert (A2 <= pawn_b && pawn_b < A8);
	assert (pw[2] == NOSQUARE && pb[2] == NOSQUARE);

	/*pp_putanchorfirst (pawn_a, pawn_b, &anchor, &loosen);*/
	anchor = pawn_a;
	loosen = pawn_b;

	if ((anchor & 07) > 3) { /* column is more than 3. e.g. = e,f,g, or h */
		anchor = flipWE (anchor);
		loosen = flipWE (loosen);
		wk     = flipWE (wk);		
		bk     = flipWE (bk);		
	}
 
	m = (SQUARE)wsq_to_pidx24 (anchor);
	n = loosen - 8;

	pp_slice = m * 48 + n; 

	if (IDX_is_empty(pp_slice)) {
		*out = NOINDEX;
		return FALSE;
	}

	assert (pp_slice < MAX_PpINDEX );
	
	*out = (index_t) (pp_slice * BLOCK_A + wk * BLOCK_B  + bk);

	return TRUE;
}


static void
kpkp_indextopc (index_t i, SQUARE *pw, SQUARE *pb)
{
	/*---------------------------------------------------------*
		inverse work to make sure that the following is valid
		index = a * BLOCK_A + b * BLOCK_B + c; 
	*----------------------------------------------------------*/
	enum  {B11100  = 7u << 2};
	enum  {BLOCK_A = 64*64, BLOCK_B = 64}; 
	enum  {block_m = 48};
	index_t a, b, c, r;
	index_t m, n;
	SQUARE sq_m, sq_n;
	
	r = i;
	a  = r / BLOCK_A;
	r -= a * BLOCK_A;
	b  = r / BLOCK_B;
	r -= b * BLOCK_B;	
	c  = r;
	
	/* unpack a, which is pslice, into m and n */
	r = a;
	m  = r / block_m;
	r -= m * block_m;
	n  = r ;

	sq_m  = pidx24_to_wsq (m);
	sq_n  = (SQUARE)n + 8;
	
	pw[0] = (SQUARE)b;
	pb[0] = (SQUARE)c;	
	pw[1] = sq_m;
	pb[1] = sq_n;
	pw[2] = NOSQUARE;
	pb[2] = NOSQUARE;	
	
	assert (A2 <= pw[1] && pw[1] < A8);
	assert (A2 <= pb[1] && pb[1] < A8);

	return;
}


/****************************************************************************\
 *
 *
 *								DEBUG ZONE 
 *
 *
 ****************************************************************************/

#if defined(DEBUG) 
static void
print_pos (const sq_t *ws, const sq_t *bs, const pc_t *wp, const pc_t *bp)
{
	int i;
	printf ("White: ");
	for (i = 0; ws[i] != NOSQUARE; i++) {
		printf ("%s%s ", P_str[wp[i]], Square_str[ws[i]]);	
	}
	printf ("\nBlack: ");
	for (i = 0; bs[i] != NOSQUARE; i++) {
		printf ("%s%s ", P_str[bp[i]], Square_str[bs[i]]);	
	}
	printf ("\n");
}
#endif

#if defined(DEBUG) || defined(FOLLOW_EGTB)
static void
output_state (unsigned stm, const SQUARE *wSQ, const SQUARE *bSQ, 
								const SQ_CONTENT *wPC, const SQ_CONTENT *bPC)
{
	int i;
	assert (stm == WH || stm == BL);
	
	printf("\n%s to move\n", stm==WH?"White":"Black");
	printf("W: ");
	for (i = 0; wSQ[i] != NOSQUARE; i++) {
		printf("%s%s ", P_str[wPC[i]], Square_str[wSQ[i]]);
	} 				
	printf("\n");
	printf("B: ");	
	for (i = 0; bSQ[i] != NOSQUARE; i++) {
		printf("%s%s ", P_str[bPC[i]], Square_str[bSQ[i]]);
	} 
	printf("\n\n");
}
#endif

static void
list_index (void)
{
	enum  {START_GTB = 0, END_GTB = (MAX_EGKEYS)};
	int i; 
	index_t accum = 0;
	printf ("\nIndex for each GTB\n");
		printf ("%3s: %7s  %7s   %7s   %7s\n" , "i", "TB", "RAM-slice", "RAM-max", "HD-cumulative");	
	for (i = START_GTB; i < END_GTB; i++) { 
		index_t indiv_k  = egkey[i].maxindex * (index_t)sizeof(dtm_t) * 2/1024;
		accum += indiv_k;
		printf ("%3d: %7s %8luk %8luk %8luM\n", i, egkey[i].str, (long unsigned)(indiv_k/egkey[i].slice_n), 
													(long unsigned)indiv_k, (long unsigned)accum/1024/2);	
	}
	printf ("\n");	
	return;
}

/**************************************************************************************************************

 NEW_WDL

**************************************************************************************************************/

/*---------------------------------------------------------------------*\
|			WDL CACHE Implementation  ZONE
\*---------------------------------------------------------------------*/

/*
|			WDL CACHE Statics
\*---------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
static unsigned int		wdl_extract (unit_t *uarr, index_t x);
static wdl_block_t *	wdl_point_block_to_replace (void);
static void				wdl_movetotop (wdl_block_t *t);

#if 0
static bool_t			wdl_cache_init (size_t cache_mem);
static void				wdl_cache_flush (void);
static bool_t			get_WDL (tbkey_t key, unsigned side, index_t idx, unsigned int *info_out, bool_t probe_hard_flag);
#endif

static bool_t			wdl_cache_is_on (void);
static void				wdl_cache_reset_counters (void);
static void				wdl_cache_done (void);

static wdl_block_t *	wdl_point_block_to_replace (void);
static bool_t			get_WDL_from_cache (tbkey_t key, unsigned side, index_t idx, unsigned int *out);
static void				wdl_movetotop (wdl_block_t *t);
static bool_t			wdl_preload_cache (tbkey_t key, unsigned side, index_t idx);

/*--------------------------------------------------------------------------*/

/*---------------------------------------------------------------------*\
|			WDL CACHE Maintainance
\*---------------------------------------------------------------------*/


static size_t
wdl_cache_init (size_t cache_mem)
{
	unsigned int 	i;
	wdl_block_t 	*p;
	size_t 			entries_per_block;
	size_t 			max_blocks;
	size_t 			block_mem;

	if (WDL_CACHE_INITIALIZED)
		wdl_cache_done();

	entries_per_block 	= 16 * 1024;  /* fixed, needed for the compression schemes */

	WDL_units_per_block	= entries_per_block / WDL_entries_per_unit;
	block_mem			= WDL_units_per_block * sizeof(unit_t);

	max_blocks 			= cache_mem / block_mem;
	cache_mem 			= max_blocks * block_mem;


	wdl_cache_reset_counters ();

	wdl_cache.entries_per_block = entries_per_block;
	wdl_cache.max_blocks 		= max_blocks;
	wdl_cache.cached 			= TRUE;
	wdl_cache.top 				= NULL;
	wdl_cache.bot 				= NULL;
	wdl_cache.n 				= 0;

	if (0 == cache_mem || NULL == (wdl_cache.buffer = (unit_t *) malloc (cache_mem))) {
		wdl_cache.cached = FALSE;
		return 0;
	}

	if (0 == max_blocks|| NULL == (wdl_cache.blocks = (wdl_block_t *) malloc (max_blocks * sizeof(wdl_block_t)))) {
		wdl_cache.cached = FALSE;
		free (wdl_cache.buffer);
		return 0;
	}
	
	for (i = 0; i < max_blocks; i++) {
		p = &wdl_cache.blocks[i];
		p->key  	= -1;
		p->side 	= gtbNOSIDE;
		p->offset 	= gtbNOINDEX;
		p->p_arr 	= wdl_cache.buffer + i * WDL_units_per_block;
		p->prev 	= NULL;
		p->next 	= NULL;
	}

	WDL_CACHE_INITIALIZED = TRUE;

	return cache_mem;
}


static void
wdl_cache_done (void)
{
	assert(WDL_CACHE_INITIALIZED);

	wdl_cache.cached = FALSE;
	wdl_cache.hard = 0;
	wdl_cache.soft = 0;
	wdl_cache.hardmisses = 0;
	wdl_cache.hits = 0;
	wdl_cache.softmisses = 0;
	wdl_cache.comparisons = 0;
	wdl_cache.max_blocks = 0;
	wdl_cache.entries_per_block = 0;

	wdl_cache.top = NULL;
	wdl_cache.bot = NULL;
	wdl_cache.n = 0;

	if (wdl_cache.buffer != NULL)
		free (wdl_cache.buffer);
	wdl_cache.buffer = NULL;

	if (wdl_cache.blocks != NULL)
		free (wdl_cache.blocks);
	wdl_cache.blocks = NULL;

	WDL_CACHE_INITIALIZED = FALSE;
	return;
}


static void
wdl_cache_flush (void)
{
	unsigned int 	i;
	wdl_block_t 	*p;
	size_t max_blocks = wdl_cache.max_blocks;

	wdl_cache.top 				= NULL;
	wdl_cache.bot 				= NULL;
	wdl_cache.n 				= 0;
	
	for (i = 0; i < max_blocks; i++) {
		p = &wdl_cache.blocks[i];
		p->key  	= -1;
		p->side 	= gtbNOSIDE;
		p->offset 	= gtbNOINDEX;
		p->p_arr 	= wdl_cache.buffer + i * WDL_units_per_block;
		p->prev 	= NULL;
		p->next 	= NULL;
	}

	wdl_cache_reset_counters  ();

	return;
}


static void
wdl_cache_reset_counters (void)
{
	wdl_cache.hard = 0;
	wdl_cache.soft = 0;
	wdl_cache.hardmisses = 0;
	wdl_cache.hits = 0;
	wdl_cache.softmisses = 0;
	wdl_cache.comparisons = 0;
	return;
}


static bool_t
wdl_cache_is_on (void)
{
	return wdl_cache.cached;
}

/****************************************************************************\
|						Replacement
\****************************************************************************/

static wdl_block_t *
wdl_point_block_to_replace (void)
{
	wdl_block_t *p, *t, *s;

	assert (0 == wdl_cache.n || wdl_cache.top != NULL);
	assert (0 == wdl_cache.n || wdl_cache.bot != NULL);
	assert (0 == wdl_cache.n || wdl_cache.bot->prev == NULL);
	assert (0 == wdl_cache.n || wdl_cache.top->next == NULL);

	if (wdl_cache.n > 0 && -1 == wdl_cache.top->key) {

		/* top blocks is unusable, should be the one to replace*/
		p = wdl_cache.top;

	} else
	if (wdl_cache.n == 0) {
		
		p = &wdl_cache.blocks[wdl_cache.n++];
		wdl_cache.top = p;
		wdl_cache.bot = p;
	
		p->prev = NULL;
		p->next = NULL;

	} else
	if (wdl_cache.n < wdl_cache.max_blocks) { /* add */

		s = wdl_cache.top;
		p = &wdl_cache.blocks[wdl_cache.n++];
		wdl_cache.top = p;
	
		s->next = p;
		p->prev = s;
		p->next = NULL;

	} else {                       /* replace*/ 
		
		t = wdl_cache.bot;
		s = wdl_cache.top;
		wdl_cache.bot = t->next;
		wdl_cache.top = t;
		
		s->next = t;
		t->prev = s;
		wdl_cache.top->next = NULL;
		wdl_cache.bot->prev = NULL;

		p = t;
	}
	
	/* make the information content unusable, it will be replaced */
	p->key    = -1;
	p->side   = gtbNOSIDE;
	p->offset = gtbNOINDEX;

	return p;
}

/****************************************************************************\
|
|						NEW PROBING ZONE
|
\****************************************************************************/

static unsigned int	wdl_extract (unit_t *uarr, index_t x);
static bool_t		get_WDL_from_cache (tbkey_t key, unsigned side, index_t idx, unsigned int *info_out);
static unsigned 	dtm2WDL(dtm_t dtm);	
static void			wdl_movetotop (wdl_block_t *t);
static bool_t		wdl_preload_cache (tbkey_t key, unsigned side, index_t idx);
static void			dtm_block_2_wdl_block(dtm_block_t *g, wdl_block_t *w, size_t n);	

static bool_t
get_WDL (tbkey_t key, unsigned side, index_t idx, unsigned int *info_out, bool_t probe_hard_flag)
{
	dtm_t dtm;
	bool_t found;

	found = get_WDL_from_cache (key, side, idx, info_out);

	if (found) {
		wdl_cache.hits++;
	} else {
		/* may probe soft */
		found = get_dtm (key, side, idx, &dtm, probe_hard_flag);
		if (found) {
			*info_out = dtm2WDL(dtm);			
			/* move cache info from dtm_cache to WDL_cache */
			if (wdl_cache_is_on())
				wdl_preload_cache (key, side, idx);
		} 
	}

	if (probe_hard_flag) {
		wdl_cache.hard++;
		if (!found) {
			wdl_cache.hardmisses++;
		}
	} else {
		wdl_cache.soft++;
		if (!found) {
			wdl_cache.softmisses++;
		}
	}

	return found;
}

static bool_t
get_WDL_from_cache (tbkey_t key, unsigned side, index_t idx, unsigned int *out)
{
	index_t 	offset;
	index_t		remainder;
	wdl_block_t	*p;
	wdl_block_t	*ret;

	if (!wdl_cache_is_on())
		return FALSE;

	split_index (wdl_cache.entries_per_block, idx, &offset, &remainder); 

	ret = NULL;
	for (p = wdl_cache.top; p != NULL; p = p->prev) {

		wdl_cache.comparisons++;

		if (key == p->key && side == p->side && offset  == p->offset) {
			ret = p;
			break;
		}
	}

	if (ret != NULL) {
		*out = wdl_extract (ret->p_arr, remainder); 
		wdl_movetotop(ret);
	}

	FOLLOW_LU("get_wdl_from_cache ok?",(ret != NULL))

	return ret != NULL;
}

static unsigned int
wdl_extract (unit_t *uarr, index_t x)
{
	index_t width = 2;
	index_t nu = x/WDL_entries_per_unit;
	index_t y  = x - (nu * WDL_entries_per_unit);
	return (uarr[nu] >> (y*width)) & WDL_entry_mask;
}

static void
wdl_movetotop (wdl_block_t *t)
{
	wdl_block_t *s, *nx, *pv;

	assert (t != NULL);

	if (t->next == NULL) /* at the top already */
		return;

	/* detach */
	pv = t->prev;
	nx = t->next;

	if (pv == NULL)  /* at the bottom */
		wdl_cache.bot = nx;
	else 
		pv->next = nx;

	if (nx == NULL) /* at the top */
		wdl_cache.top = pv;
	else
		nx->prev = pv;

	/* relocate */
	s = wdl_cache.top;
	assert (s != NULL);
	if (s == NULL)
		wdl_cache.bot = t;	
	else
		s->next = t;

	t->next = NULL;
	t->prev = s;
	wdl_cache.top = t;

	return;
}

/****************************************************************************************************/

static bool_t
wdl_preload_cache (tbkey_t key, unsigned side, index_t idx)
/* output to the least used block of the cache */
{
	dtm_block_t		*dtm_block;
	wdl_block_t 	*to_modify;
	bool_t 			ok;

	FOLLOW_label("wdl preload_cache starts")

	if (idx >= egkey[key].maxindex) {
		FOLLOW_LULU("Wrong index", __LINE__, idx)	
		return FALSE;
	}

	/* find fresh block in dtm cache */
	dtm_block = dtm_cache_pointblock (key, side, idx); 

	/* find aged blocked in wdl cache */
	to_modify = wdl_point_block_to_replace ();

	ok = !(NULL == dtm_block || NULL == to_modify);

	if (!ok)
		return FALSE;
	
	/* transform and move a block */
	dtm_block_2_wdl_block(dtm_block, to_modify, dtm_cache.entries_per_block);	

	if (ok) {
		index_t 		offset;
		index_t			remainder;
		split_index (wdl_cache.entries_per_block, idx, &offset, &remainder); 

		to_modify->key    = key;
		to_modify->side   = side;
		to_modify->offset = offset;
	} else {
		/* make it unusable */
		to_modify->key    = -1;
		to_modify->side   = gtbNOSIDE;
		to_modify->offset = gtbNOINDEX;
	}

	FOLLOW_LU("wdl preload_cache?", ok)

	return ok;		
}

/****************************************************************************************************/

static void			
dtm_block_2_wdl_block(dtm_block_t *g, wdl_block_t *w, size_t n)
{
	int width = 2;
	int shifting;
	size_t i;
	int j;
	unsigned int x ,y;
	 dtm_t *s = g->p_arr;
	unit_t *d = w->p_arr;

	for (i = 0, y = 0; i < n; i++) {
		j =  i & 3; /* modulo WDL_entries_per_unit */
		x = dtm2WDL(s[i]);
		shifting = j * width;
		y |= (x << shifting);		
		
		if (j == 3) {
			d[i/WDL_entries_per_unit] = (unit_t) y;
			y = 0;
		}
	}

	if (0 != (n & 3)) { /* not multiple of 4 */
		d[(n-1)/WDL_entries_per_unit] = (unit_t) y; /* save the rest     */
		y = 0;
	}

	return;
}	

static unsigned 	
dtm2WDL(dtm_t dtm)
{
	return (unsigned) dtm & 3;
}	


/**************************/
#ifdef WDL_PROBE

static unsigned int	inv_wdl(unsigned w);
static bool_t	egtb_get_wdl (tbkey_t k, unsigned stm, const SQUARE *wS, const SQUARE *bS, bool_t probe_hard_flag, unsigned int *wdl);

static bool_t
tb_probe_wdl
			(unsigned stm, 
			 const SQUARE *inp_wSQ, 
			 const SQUARE *inp_bSQ,
			 const SQ_CONTENT *inp_wPC, 
			 const SQ_CONTENT *inp_bPC,
			 bool_t probingtype,
			 /*@out@*/ unsigned *res)
{
	tbkey_t id = -1;
	unsigned int wdl = iUNKNOWN;

	SQUARE 		storage_ws [MAX_LISTSIZE], storage_bs [MAX_LISTSIZE];
	SQ_CONTENT  storage_wp [MAX_LISTSIZE], storage_bp [MAX_LISTSIZE];

	SQUARE     *ws = storage_ws;
	SQUARE     *bs = storage_bs;
	SQ_CONTENT *wp = storage_wp;
	SQ_CONTENT *bp = storage_bp;
	SQUARE 		tmp_ws [MAX_LISTSIZE], tmp_bs [MAX_LISTSIZE];
	SQ_CONTENT  tmp_wp [MAX_LISTSIZE], tmp_bp [MAX_LISTSIZE];

	SQUARE *temps;
	bool_t straight = FALSE;
	
	bool_t  okcall  = TRUE;
	unsigned ply_;
	unsigned *ply = &ply_;

	/************************************/

	assert (stm == WH || stm == BL);

	/* VALID ONLY FOR KK!! */
	if (inp_wPC[1] == NOPIECE && inp_bPC[1] == NOPIECE) {
		index_t dummy_i;
		bool_t b = kxk_pctoindex (inp_wSQ, inp_bSQ, &dummy_i);
		*res = b? iDRAW: iFORBID;
		*ply = 0;
		return TRUE;
	} 

	/* copy input */
	list_pc_copy (inp_wPC, wp);
	list_pc_copy (inp_bPC, bp);
	list_sq_copy (inp_wSQ, ws);
	list_sq_copy (inp_bSQ, bs);

	sortlists (ws, wp);
	sortlists (bs, bp);

	FOLLOW_label("EGTB_PROBE")

	if (egtb_get_id (wp, bp, &id)) {
		FOLLOW_LU("got ID",id)
		straight = TRUE;
	} else if (egtb_get_id (bp, wp, &id)) {
		FOLLOW_LU("rev ID",id)
		straight = FALSE;
		list_sq_flipNS (ws);
		list_sq_flipNS (bs);
        temps = ws;
        ws = bs;
        bs = temps;
		stm = Opp(stm);
		/* no enpassant in this fuction, so no adjustment */
		{SQ_CONTENT *tempp = wp; wp = bp; bp = tempp;} 	/* added */
	} else {
		#if defined(DEBUG)
		printf("did not get id...\n");
		output_state (stm, ws, bs, wp, bp);		
		#endif
		unpackdist (iFORBID, res, ply);
		return FALSE;
	}

	/* store position... */
	list_pc_copy (wp, tmp_wp);
	list_pc_copy (bp, tmp_bp);
	list_sq_copy (ws, tmp_ws);
	list_sq_copy (bs, tmp_bs);

	/* x will be stm and y will be stw */
/*
	if (stm == WH) {
        xs = ws;
        xp = wp;
        ys = bs;
        yp = bp;
    } else {
        xs = bs;
        xp = bp;
        ys = ws;
        yp = wp;
	}
*/
	okcall = egtb_get_wdl (id, stm, ws, bs, probingtype, &wdl);

	FOLLOW_LU("dtmok?",okcall)
	FOLLOW_DTM("wdl", wdl)

	if (okcall) {

		/*assert(epsq == NOSQUARE); */

		if (straight) {
			*res = wdl;
		} else {
			*res = inv_wdl (wdl);
		}	 
	} else {
			unpackdist (iFORBID, res, ply);
	}

	return okcall;
} 

static unsigned int
inv_wdl(unsigned w)
{
	unsigned r = tb_UNKNOWN;
	switch (w) {
		case tb_DRAW:    r = tb_DRAW;    break;
		case tb_WMATE:   r = tb_BMATE;   break;
		case tb_BMATE:   r = tb_WMATE;   break;
		case tb_FORBID:  r = tb_FORBID;  break;
		case tb_UNKNOWN: r = tb_UNKNOWN; break;
		default:         r = tb_UNKNOWN; break;
	}
	return r;
}

static bool_t
egtb_get_wdl (tbkey_t k, unsigned stm, const SQUARE *wS, const SQUARE *bS, bool_t probe_hard_flag, unsigned int *wdl)
{
	bool_t idxavail;
	index_t idx;
	dtm_t *tab[2];
	bool_t (*pc2idx) (const SQUARE *, const SQUARE *, index_t *);

	FOLLOW_label("egtb_get_wdl --> starts")

	if (egkey[k].status == STATUS_MALLOC || egkey[k].status == STATUS_STATICRAM) {

		tab[WH] = egkey[k].egt_w;
		tab[BL] = egkey[k].egt_b;
		pc2idx  = egkey[k].pctoi;

		idxavail = pc2idx (wS, bS, &idx);

		FOLLOW_LU("indexavail (RAM)",idxavail)

		if (idxavail) {
			*wdl = dtm2WDL(tab[stm][idx]);
		} else {
			*wdl = dtm2WDL(iFORBID);
		}

		return FALSE;

	} else if (egkey[k].status == STATUS_ABSENT) {

		pc2idx   = egkey[k].pctoi;
		idxavail = pc2idx (wS, bS, &idx);

		FOLLOW_LU("indexavail (HD)",idxavail)

		if (idxavail) {
			bool_t success;

			/* 
			|		LOCK 
			*-------------------------------*/
			mythread_mutex_lock (&Egtb_lock);	

			success = get_WDL (k, stm, idx, wdl, probe_hard_flag);
			FOLLOW_LU("get_wld (succ)",success)
			FOLLOW_LU("get_wld (wdl )",*wdl)

			/* this may not be needed */
			if (!success) {		
				dtm_t dtm;
				unsigned res, ply;
				if (probe_hard_flag && Uncompressed) {
					assert(Uncompressed);
					success = egtb_filepeek (k, stm, idx, &dtm);
					unpackdist (dtm, &res, &ply);			
					*wdl = res;		
				}
				else
					success = FALSE;
			}

			mythread_mutex_unlock (&Egtb_lock);	
			/*------------------------------*\ 
			|		UNLOCK 
			*/

			if (success) {
				return TRUE;
			} else {
				if (probe_hard_flag) /* after probing hard and failing, no chance to succeed later */
					egkey[k].status = STATUS_REJECT;
				*wdl = dtm2WDL(iUNKNOWN);
				return FALSE;
			}

		} else {
			*wdl = dtm2WDL(iFORBID);
			return 	TRUE;
		}
	} else if (egkey[k].status == STATUS_REJECT) {
		FOLLOW_label("STATUS_REJECT")
		*wdl = dtm2WDL(iFORBID);
		return 	FALSE;
	} else {
		FOLLOW_label("STATUS_WRONG!")
		assert(0);
		*wdl = dtm2WDL(iFORBID);
		return 	FALSE;
	} 

} 
#endif



