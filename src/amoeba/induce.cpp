#include "ff/amoebamod.h"
#include "ff/atom.h"
#include "ff/nblist.h"
#include "ff/potent.h"
#include "tool/externfunc.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polar.hh>
#include <tinker/detail/units.hh>

namespace tinker {
TINKER_FVOID2(cu, 0, acc, 1, diagPrecond, const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3]);
void diagPrecond(const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3])
{
   TINKER_FCALL2(cu, 0, acc, 1, diagPrecond, rsd, rsdp, zrsd, zrsdp);
}

void sparsePrecondBuild() {}

TINKER_FVOID2(cu, 1, acc, 1, sparsePrecondApply, const real (*)[3], const real (*)[3], real (*)[3],
   real (*)[3]);
void sparsePrecondApply(
   const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3])
{
   TINKER_FCALL2(cu, 1, acc, 1, sparsePrecondApply, rsd, rsdp, zrsd, zrsdp);
}

TINKER_FVOID2(cu, 1, acc, 1, ulspredSaveP1, real (*)[3], real (*)[3], //
   const real (*)[3], const real (*)[3]);
void ulspredSave(const real (*uind)[3], const real (*uinp)[3])
{
   if (polpred == UPred::NONE)
      return;

   // clang-format off
   real(*ud)[3];
   real(*up)[3];
   int pos = nualt % maxualt;
   switch (pos) {
      case  0: ud = udalt_00; up = upalt_00; break;
      case  1: ud = udalt_01; up = upalt_01; break;
      case  2: ud = udalt_02; up = upalt_02; break;
      case  3: ud = udalt_03; up = upalt_03; break;
      case  4: ud = udalt_04; up = upalt_04; break;
      case  5: ud = udalt_05; up = upalt_05; break;
      case  6: ud = udalt_06; up = upalt_06; break;
      case  7: ud = udalt_07; up = upalt_07; break;
      case  8: ud = udalt_08; up = upalt_08; break;
      case  9: ud = udalt_09; up = upalt_09; break;
      case 10: ud = udalt_10; up = upalt_10; break;
      case 11: ud = udalt_11; up = upalt_11; break;
      case 12: ud = udalt_12; up = upalt_12; break;
      case 13: ud = udalt_13; up = upalt_13; break;
      case 14: ud = udalt_14; up = upalt_14; break;
      case 15: ud = udalt_15; up = upalt_15; break;
      default: ud =  nullptr; up =  nullptr; break;
   }
   nualt = nualt + 1;
   // clang-format on
   if (nualt > 2 * maxualt)
      nualt = nualt - maxualt;

   TINKER_FCALL2(cu, 1, acc, 1, ulspredSaveP1, ud, up, uind, uinp);
}

TINKER_FVOID2(cu, 0, acc, 1, ulspredSum, real (*)[3], real (*)[3]);
void ulspredSum(real (*uind)[3], real (*uinp)[3])
{
   TINKER_FCALL2(cu, 0, acc, 1, ulspredSum, uind, uinp);
}
}

namespace tinker {
TINKER_FVOID2(cu, 1, acc, 1, induceMutualPcg1, real (*)[3], real (*)[3]);
static void induceMutualPcg1(real (*uind)[3], real (*uinp)[3])
{
   TINKER_FCALL2(cu, 1, acc, 1, induceMutualPcg1, uind, uinp);
}

void inducePrint(const real (*ud)[3])
{
   if (inform::debug and use(Potent::POLAR)) {
      std::vector<double> uindbuf;
      uindbuf.resize(3 * n);
      darray::copyout(g::q0, n, uindbuf.data(), ud);
      waitFor(g::q0);
      bool header = true;
      for (int i = 0; i < n; ++i) {
         if (polar::polarity[i] != 0) {
            if (header) {
               header = false;
               print(stdout, "\n Induced Dipole Moments (Debye) :\n");
               print(stdout, "\n    Atom %1$13s X %1$10s Y %1$10s Z %1$9s Total\n\n", "");
            }
            double u1 = uindbuf[3 * i];
            double u2 = uindbuf[3 * i + 1];
            double u3 = uindbuf[3 * i + 2];
            double unorm = std::sqrt(u1 * u1 + u2 * u2 + u3 * u3);
            u1 *= units::debye;
            u2 *= units::debye;
            u3 *= units::debye;
            unorm *= units::debye;
            print(stdout, "%8d     %13.4f%13.4f%13.4f %13.4f\n", i + 1, u1, u2, u3, unorm);
         }
      }
   }
}

void induce(real (*ud)[3], real (*up)[3])
{
   induceMutualPcg1(ud, up);
   ulspredSave(ud, up);
   inducePrint(ud);
}
}
