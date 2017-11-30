#ifndef DMM_H
   #define DMM_H

   #define ATL_mmMULADD
   #define ATL_mmLAT 5
   #define ATL_mmMU  4
   #define ATL_mmNU  1
   #define ATL_mmKU  60
   #define MB 60
   #define NB 60
   #define KB 60
   #define NBNB 3600
   #define MBNB 3600
   #define MBKB 3600
   #define NBKB 3600
   #define NB2 120
   #define NBNB2 7200

   #define ATL_MulByNB(N_) ((N_) * 60)
   #define ATL_DivByNB(N_) ((N_) / 60)
   #define ATL_MulByNBNB(N_) ((N_) * 3600)
   #define NBmm ATL_dJIK60x60x60TN60x60x0_a1_b1
   #define NBmm_b1 ATL_dJIK60x60x60TN60x60x0_a1_b1
   #define NBmm_b0 ATL_dJIK60x60x60TN60x60x0_a1_b0
   #define NBmm_bX ATL_dJIK60x60x60TN60x60x0_a1_bX

#endif
