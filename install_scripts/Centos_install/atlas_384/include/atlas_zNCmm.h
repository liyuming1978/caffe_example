#ifndef ZMM_H
   #define ZMM_H

   #define ATL_mmMULADD
   #define ATL_mmLAT 5
   #define ATL_mmMU  5
   #define ATL_mmNU  1
   #define ATL_mmKU  60
   #define MB 20
   #define NB 20
   #define KB 20
   #define NBNB 400
   #define MBNB 400
   #define MBKB 400
   #define NBKB 400
   #define NB2 40
   #define NBNB2 800

   #define ATL_MulByNB(N_) ((N_) * 20)
   #define ATL_DivByNB(N_) ((N_) / 20)
   #define ATL_MulByNBNB(N_) ((N_) * 400)

#endif
