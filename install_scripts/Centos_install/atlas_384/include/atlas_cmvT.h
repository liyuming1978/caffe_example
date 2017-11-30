#ifndef ATLAS_MVT_H
#define ATLAS_MVT_H

#include "atlas_misc.h"
#include "atlas_cNCmm.h"
#include "atlas_lvl3.h"

#define ATL_mvTMU ATL_mmMU
#define ATL_mvTNU NB
#define ATL_mvTCallsGemm
#ifndef ATL_L1mvelts
   #define ATL_L1mvelts ((3*ATL_L1elts)>>2)
#endif

#define ATL_GetPartMVT(A_, lda_, mb_, nb_) \
{ \
   *(mb_) = (ATL_L1mvelts - NB - ATL_mmMU*(NB+1)) / (NB+1); \
   if (*(mb_) > NB) \
   { \
      *(mb_) = ATL_MulByNB(ATL_DivByNB(*(mb_))); \
      *(nb_) = NB; \
   } \
   else \
   { \
      if (*(mb_) > ATL_mmMU) *(nb_) = (*(mb_) / ATL_mmMU)*ATL_mmMU; \
      else *(nb_) = ATL_mmMU; \
      *(mb_) = NB; \
   } \
}

#endif
