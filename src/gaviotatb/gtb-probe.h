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

#if !defined(H_GTBPROBE)
#define H_GTBPROBE
#ifdef __cplusplus
extern "C" {
#endif
/*>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/

#include <stdlib.h>

#define tb_MAXPATHLEN 1024

/*----------------------------------*\
|            CONSTANTS
\*----------------------------------*/

enum TB_mask_values { tb_RESMASK = 3, tb_INFOMASK = 7, tb_PLYSHIFT = 3 };

enum TB_return_values {   
   tb_DRAW = 0,
   tb_WMATE = 1,
   tb_BMATE = 2,
   tb_FORBID = 3,
   tb_UNKNOWN = 7
};

enum TB_pieces    {
   tb_NOPIECE, tb_PAWN, tb_KNIGHT, tb_BISHOP, tb_ROOK, tb_QUEEN, tb_KING
};

enum TB_sides {
   tb_WHITE_TO_MOVE, tb_BLACK_TO_MOVE
};

enum TB_squares {
   tb_A1, tb_B1, tb_C1, tb_D1, tb_E1, tb_F1, tb_G1, tb_H1,
   tb_A2, tb_B2, tb_C2, tb_D2, tb_E2, tb_F2, tb_G2, tb_H2,
   tb_A3, tb_B3, tb_C3, tb_D3, tb_E3, tb_F3, tb_G3, tb_H3,
   tb_A4, tb_B4, tb_C4, tb_D4, tb_E4, tb_F4, tb_G4, tb_H4,
   tb_A5, tb_B5, tb_C5, tb_D5, tb_E5, tb_F5, tb_G5, tb_H5,
   tb_A6, tb_B6, tb_C6, tb_D6, tb_E6, tb_F6, tb_G6, tb_H6,
   tb_A7, tb_B7, tb_C7, tb_D7, tb_E7, tb_F7, tb_G7, tb_H7,
   tb_A8, tb_B8, tb_C8, tb_D8, tb_E8, tb_F8, tb_G8, tb_H8,
   tb_NOSQUARE
};

enum TB_castling {
   tb_NOCASTLE = 0,
   tb_WOO  = 8,
   tb_WOOO = 4,    
   tb_BOO  = 2,
   tb_BOOO = 1
};

enum TB_compression_scheme {
	tb_UNCOMPRESSED, tb_CP1, tb_CP2, tb_CP3, tb_CP4 
};

/*----------------------------------*\
|         	FUNCTIONS
\*----------------------------------*/

extern char *		tb_init   (int verbosity, int compression_scheme, const char **paths);

extern char *		tb_restart(int verbosity, int compression_scheme, const char **paths);

extern void			tb_done (void);

extern int /*bool*/	tb_probe_hard 
							(unsigned stm, 
			 				 unsigned epsq,
							 unsigned castles,
			 				 const unsigned *inp_wSQ, 
			 				 const unsigned *inp_bSQ,
			 				 const unsigned char *inp_wPC, 
			 				 const unsigned char *inp_bPC,
			 				 /*@out@*/ unsigned *tbinfo, 
			 				 /*@out@*/ unsigned *plies);

extern int /*bool*/	tb_probe_soft 
							(unsigned stm, 
			 				 unsigned epsq,
							 unsigned castles,
			 				 const unsigned *inp_wSQ, 
			 				 const unsigned *inp_bSQ,
			 				 const unsigned char *inp_wPC, 
			 				 const unsigned char *inp_bPC,
			 				 /*@out@*/ unsigned *tbinfo, 
			 				 /*@out@*/ unsigned *plies);

extern int /*bool*/	tb_probe_WDL_hard 
							(unsigned stm, 
			 				 unsigned epsq,
							 unsigned castles,
			 				 const unsigned *inp_wSQ, 
			 				 const unsigned *inp_bSQ,
			 				 const unsigned char *inp_wPC, 
			 				 const unsigned char *inp_bPC,
			 				 /*@out@*/ unsigned *tbinfo); 

extern int /*bool*/	tb_probe_WDL_soft 
							(unsigned stm, 
			 				 unsigned epsq,
							 unsigned castles,
			 				 const unsigned *inp_wSQ, 
			 				 const unsigned *inp_bSQ,
			 				 const unsigned char *inp_wPC, 
			 				 const unsigned char *inp_bPC,
			 				 /*@out@*/ unsigned *tbinfo);

extern int /*bool*/	tb_is_initialized (void);

/* 	
|	tb_availability() returns 0 if no TBs available 
|	Otherwise, the following bits are turned 'on' if...
|	----------------------------------
|	bit: meaning
|	----------------------------------
|	0: at least one 3-pc TB is present
|	1: 3-pc TBs are complete
|	2: at least one 4-pc TB is present
|	3: 4-pc TBs are complete
|	4: at least one 5-pc TB is present
|	5: 5-pc TBs are complete
|	6: at least one 6-pc TB is present
|	7: 6-pc TBs are complete
|	Example: if 63 is returned, it means all 3-4-5-pc TBs are present
|	Bits 6 and 7 will be always off, of course, until 6-piece TBs
|	are supported.
*/

extern unsigned int	tb_availability(void);

/* 	
|	tb_indexmemory ()returns 
|	how much memory has been allocated for indexes 
*/

extern size_t		tb_indexmemory (void);

/*----------------------------------*\
|         	cache
\*----------------------------------*/

extern int /*bool*/	tbcache_init    (size_t cache_mem, int wdl_fraction);
extern int /*bool*/	tbcache_restart (size_t cache_mem, int wdl_fraction);
extern void			tbcache_done (void);
extern int /*bool*/	tbcache_is_on (void);
extern void			tbcache_flush (void);

/*----------------------------------*\
|         	STATS
\*----------------------------------*/

/*
|	For maximum portability, some stats are provided
|	in two 32 bits integers rather than a single 64 bit number. 
|	For intance, prob_hard_hits[0] contains the less significant 32 bits 
|	(0 to 31), and prob_hard_hits[1] the most significant ones (32 to 63). 
|	The 64-bit number can be recreated like this:
|	uint64_t x;
|	x = (uint64_t)probe_hard_hits[0]|((uint64_t)probe_hard_hits[1]<<32);
|	The user has the responsibility to combine the numbers and use the 
|	proper 64 bit integers.
*/

struct TB_STATS {
	long unsigned int wdl_easy_hits  [2]; /* hits that were found in own wdl cache */
	long unsigned int wdl_hard_prob  [2]; /* hard probes to the wdl cache: if fail, they will go to HD */
	long unsigned int wdl_soft_prob  [2]; /* soft probes to the wdl cache: if fail, they won't go to HD */	
	size_t			  wdl_cachesize     ; /* size allocated for wdl cache */
	double			  wdl_occupancy     ; /* % of slots filled in wdl cache */

	long unsigned int dtm_easy_hits  [2]; /* hits that were found in own dtm cache */
	long unsigned int dtm_hard_prob  [2]; /* hard probes to the dtm cache: if fail, they will go to HD */
	long unsigned int dtm_soft_prob  [2]; /* soft probes to the dtm cache: if fail, they won't go to HD */	
	size_t			  dtm_cachesize     ; /* size allocated for dtm cache */
	double			  dtm_occupancy     ; /* % of slots filled in dtm cache */

	long unsigned int  total_hits    [2]; /* succesful probes */
	long unsigned int memory_hits    [2]; /* succesful probes to memory */
	long unsigned int  drive_hits    [2]; /* succesful probes to the Hard drive */
	long unsigned int  drive_miss    [2]; /* failing   probes to the Hard drive */
	long unsigned int  bytes_read    [2]; /* bytes read from Hard drive */
	long unsigned int files_opened      ; /* number of files newly opened */
	double			  memory_efficiency ; /* % hits from memory over total hits */
};

extern void			tbstats_reset (void);
extern void 		tbstats_get (struct TB_STATS *stats);


/*----------------------------------*\
|         	PATH MANAGEMENT
\*----------------------------------*/

extern const char ** 		tbpaths_init	(void);
extern const char ** 		tbpaths_add		(const char **ps, const char *newpath);
extern const char ** 		tbpaths_done	(const char **ps);

extern const char *			tbpaths_getmain (void);

/*<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/
#ifdef __cplusplus
}
#endif
#endif
