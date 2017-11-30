#ifndef SMM_H
   #define SMM_H

   #define ATL_mmMULADD
   #define ATL_mmLAT 5
   #define ATL_mmMU  5
   #define ATL_mmNU  1
   #define ATL_mmKU  84
   #define MB 84
   #define NB 84
   #define KB 84
   #define NBNB 7056
   #define MBNB 7056
   #define MBKB 7056
   #define NBKB 7056
   #define NB2 168
   #define NBNB2 14112

   #define ATL_MulByNB(N_) ((N_) * 84)
   #define ATL_DivByNB(N_) ((N_) / 84)
   #define ATL_MulByNBNB(N_) ((N_) * 7056)
   #define NBmm ATL_sJIK84x84x84TN84x84x0_a1_b1
   #define NBmm_b1 ATL_sJIK84x84x84TN84x84x0_a1_b1
   #define NBmm_b0 ATL_sJIK84x84x84TN84x84x0_a1_b0
   #define NBmm_bX ATL_sJIK84x84x84TN84x84x0_a1_bX

#endif
