#ifndef CMM_H
   #define CMM_H

   #define ATL_mmMULADD
   #define ATL_mmLAT 1
   #define ATL_mmMU  5
   #define ATL_mmNU  1
   #define ATL_mmKU  84
   #define MB 28
   #define NB 28
   #define KB 28
   #define NBNB 784
   #define MBNB 784
   #define MBKB 784
   #define NBKB 784
   #define NB2 56
   #define NBNB2 1568

   #define ATL_MulByNB(N_) ((N_) * 28)
   #define ATL_DivByNB(N_) ((N_) / 28)
   #define ATL_MulByNBNB(N_) ((N_) * 784)

#endif
