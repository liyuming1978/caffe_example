#ifndef ATLAS_MVS_H
#define ATLAS_MVS_H

#include "atlas_misc.h"
#include "atlas_cNCmm.h"
#include "atlas_lvl3.h"

#define ATL_mvSMU ATL_mmMU
#define ATL_mvSNU NB
#define ATL_mvSCallsGemm
#ifndef ATL_L1mvelts
   #define ATL_L1mvelts ((3*ATL_L1elts)>>2)
#endif
#ifndef ATL_mvNNU
   #include "atlas_cmvN.h"
#endif
#ifndef ATL_mvTNU
   #include "atlas_cmvT.h"
#endif
#define ATL_GetPartSYMV(A_, lda_, mb_, nb_) *(mb_) = *(nb_) = NB 

#endif
