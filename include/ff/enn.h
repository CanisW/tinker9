#pragma once
#include "ff/energybuffer.h"
#include "tool/rcman.h"


namespace tinker {

/// Computes the nn potential energy. it can be valence or not.
void ennData(RcOp);
void enn(int vers, bool is_bonded);

TINKER_EXTERN EnergyBuffer enn_buf;
TINKER_EXTERN VirialBuffer vir_enn;
TINKER_EXTERN grad_prec* denn_x;
TINKER_EXTERN grad_prec* denn_y;
TINKER_EXTERN grad_prec* denn_z;
TINKER_EXTERN energy_prec energy_enn;
TINKER_EXTERN virial_prec virial_enn[9];

/// \}
}
