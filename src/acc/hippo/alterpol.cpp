#include "ff/amoebamod.h"
#include "ff/atom.h"
#include "ff/elec.h"
#include "ff/hippomod.h"
#include "ff/image.h"
#include "ff/nblist.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/pair_alterpol.h"
#include "tool/gpucard.h"
#include <tinker/routines.h>

namespace tinker {
void alterpol(real (*polscale)[3][3], real (*polinv)[3][3])
{
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   // initialize polscale and polinv
   #pragma acc parallel loop independent async\
               deviceptr(polscale)
   for (int i = 0; i < n; ++i) {
      polscale[i][0][0] = 1.f;
      polscale[i][0][1] = 0.f;
      polscale[i][0][2] = 0.f;
      polscale[i][1][0] = 0.f;
      polscale[i][1][1] = 1.f;
      polscale[i][1][2] = 0.f;
      polscale[i][2][0] = 0.f;
      polscale[i][2][1] = 0.f;
      polscale[i][2][2] = 1.f;
   }

   // find variable polarizability scale matrix at each site
   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(x,y,z,kpep,prepep,dmppep,lpep,mlst,polscale)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real springi = kpep[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real r2 = image2(xr, yr, zr);
         int eplk = lpep[k];
         bool incl = (epli || eplk);
         if (r2 <= off2 and incl) {
            real r = REAL_SQRT(r2);
            real springk = kpep[k];
            real sizk = prepep[k];
            real alphak = dmppep[k];
            real ks2i[3][3], ks2k[3][3];
            pair_alterpol(scrtyp, r, r2, 1, cut, off, xr, yr, zr, springi, sizi, alphai, springk,
               sizk, alphak, ks2i, ks2k);
            #pragma acc loop seq
            for (int l = 0; l < 3; ++l) {
               #pragma acc loop seq
               for (int m = 0; m < 3; ++m) {
                  polscale[i][m][l] += ks2i[m][l];
                  polscale[k][m][l] += ks2k[m][l];
               }
            }
         }
      }
   }

   #pragma acc parallel loop independent async\
               deviceptr(x,y,z,kpep,prepep,dmppep,lpep,mlst,mdwexclude,mdwexclude_scale,polscale)
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real dscale = mdwexclude_scale[ii][1] - 1;

      real xr = x[k] - x[i];
      real yr = y[k] - y[i];
      real zr = z[k] - z[i];
      real springi = kpep[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];
      real springk = kpep[k];
      real sizk = prepep[k];
      real alphak = dmppep[k];
      int eplk = lpep[k];

      real r2 = image2(xr, yr, zr);
      bool incl1 = dscale != 0;
      bool incl2 = (epli || eplk);
      if (r2 <= off2 and incl1 and incl2) {
         real r = REAL_SQRT(r2);
         real ks2i[3][3], ks2k[3][3];
         pair_alterpol(scrtyp, r, r2, dscale, cut, off, xr, yr, zr, springi, sizi, alphai, springk, sizk,
            alphak, ks2i, ks2k);
         #pragma acc loop seq
         for (int l = 0; l < 3; ++l) {
            #pragma acc loop seq
            for (int m = 0; m < 3; ++m) {
               polscale[i][m][l] = polscale[i][m][l] + ks2i[m][l];
               polscale[k][m][l] = polscale[k][m][l] + ks2k[m][l];
            }
         }
      }
   }

   // invert
   #pragma acc parallel loop independent async\
               deviceptr(polscale,polinv)
   for (int i = 0; i < n; ++i) {
      real det;
      real(&ps)[3][3] = polscale[i];
      det = ps[0][0] * (ps[1][1] * ps[2][2] - ps[1][2] * ps[2][1]) -
         ps[1][0] * (ps[0][1] * ps[2][2] - ps[2][1] * ps[0][2]) +
         ps[2][0] * (ps[0][1] * ps[1][2] - ps[1][1] * ps[0][2]);
      polinv[i][0][0] = (ps[1][1] * ps[2][2] - ps[1][2] * ps[2][1]) / det;
      polinv[i][1][0] = (ps[2][0] * ps[1][2] - ps[1][0] * ps[2][2]) / det;
      polinv[i][2][0] = (ps[1][0] * ps[2][1] - ps[2][0] * ps[1][1]) / det;
      polinv[i][0][1] = (ps[2][1] * ps[0][2] - ps[0][1] * ps[2][2]) / det;
      polinv[i][1][1] = (ps[0][0] * ps[2][2] - ps[2][0] * ps[0][2]) / det;
      polinv[i][2][1] = (ps[0][1] * ps[2][0] - ps[0][0] * ps[2][1]) / det;
      polinv[i][0][2] = (ps[0][1] * ps[1][2] - ps[0][2] * ps[1][1]) / det;
      polinv[i][1][2] = (ps[0][2] * ps[1][0] - ps[0][0] * ps[1][2]) / det;
      polinv[i][2][2] = (ps[0][0] * ps[1][1] - ps[0][1] * ps[1][0]) / det;
   }
}

void dexpol(const int vers, const real (*uind)[3], grad_prec* depx, grad_prec* depy, grad_prec* depz,
   VirialBuffer restrict vir_ep)
{
   auto do_v = vers & calc::virial;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = bufferSize();

   const real f = 0.5f * electric / dielec;

   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real springi = kpep[i]/polarity[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];
      real uix = uind[i][0];
      real uiy = uind[i][1];
      real uiz = uind[i][2];

      MAYBE_UNUSED real gxi = 0., gyi = 0., gzi = 0.;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = kk & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real r2 = image2(xr, yr, zr);
         int eplk = lpep[k];
         bool incl = (epli || eplk);
         if (r2 <= off2 and incl) {
            real r = REAL_SQRT(r2);
            real springk = kpep[k]/polarity[k];
            real sizk = prepep[k];
            real alphak = dmppep[k];
            real ukx = uind[k][0];
            real uky = uind[k][1];
            real ukz = uind[k][2];
            real frc[3];
            pair_dexpol(scrtyp, r, r2, 1, cut, off, xr, yr, zr, uix, uiy, uiz, ukx, uky, ukz,
               springi, sizi, alphai, springk, sizk, alphak, f, frc);
            gxi += frc[0];
            gyi += frc[1];
            gzi += frc[2];
            atomic_add(-frc[0], depx, k);
            atomic_add(-frc[1], depy, k);
            atomic_add(-frc[2], depz, k);

            if (do_v) {
               real vxx = -xr * frc[0];
               real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
               real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
               real vyy = -yr * frc[1];
               real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
               real vzz = -zr * frc[2];
               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, offset);
            }
         }
      }
      atomic_add(gxi, depx, i);
      atomic_add(gyi, depy, i);
      atomic_add(gzi, depz, i);
   }

   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real dscale = mdwexclude_scale[ii][1] - 1;

      real xr = x[k] - x[i];
      real yr = y[k] - y[i];
      real zr = z[k] - z[i];
      real springi = kpep[i];
      real sizi = prepep[i];
      real alphai = dmppep[i];
      int epli = lpep[i];
      real uix = uind[i][0];
      real uiy = uind[i][1];
      real uiz = uind[i][2];
      real springk = kpep[k];
      real sizk = prepep[k];
      real alphak = dmppep[k];
      int eplk = lpep[k];
      real ukx = uind[k][0];
      real uky = uind[k][1];
      real ukz = uind[k][2];

      real r2 = image2(xr, yr, zr);
      bool incl1 = dscale != 0;
      bool incl2 = (epli || eplk);

      if (r2 <= off2 and incl1 and incl2) {
         real r = REAL_SQRT(r2);
         real frc[3];
         pair_dexpol(scrtyp, r, r2, dscale, cut, off, xr, yr, zr, uix, uiy, uiz, ukx, uky, ukz, springi,
            sizi, alphai, springk, sizk, alphak, f, frc);

         atomic_add(frc[0], depx, i);
         atomic_add(frc[1], depy, i);
         atomic_add(frc[2], depz, i);
         atomic_add(-frc[0], depx, k);
         atomic_add(-frc[1], depy, k);
         atomic_add(-frc[2], depz, k);
         
         if (do_v) {
            real vxx = -xr * frc[0];
            real vxy = -0.5f * (yr * frc[0] + xr * frc[1]);
            real vxz = -0.5f * (zr * frc[0] + xr * frc[2]);
            real vyy = -yr * frc[1];
            real vyz = -0.5f * (zr * frc[1] + yr * frc[2]);
            real vzz = -zr * frc[2];
            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_ep, offset);
         }
      }
   }
}
}
