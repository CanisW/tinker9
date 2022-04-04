#include "ff/amoeba/elecamoeba.h"
#include "ff/atom.h"
#include "ff/image.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/switch.h"
#include "seq/pair_field.h"
#include "tool/gpucard.h"

namespace tinker {
// see also subroutine udirect1 in induce.f
void dfieldEwaldRecipSelf_acc(real (*field)[3])
{
   darray::zero(g::q0, n, field);

   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;
   const real term = aewald * aewald * aewald * 4 / 3 / sqrtpi;

   cmpToFmp(pu, cmp, fmp);
   gridMpole(pu, fmp);
   fftfront(pu);
   if (vir_m)
      pmeConv(pu, vir_m);
   else
      pmeConv(pu);
   fftback(pu);
   fphiMpole(pu);
   fphiToCphi(pu, fphi, cphi);

   #pragma acc parallel loop independent async deviceptr(field,cphi,rpole)
   for (int i = 0; i < n; ++i) {
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      field[i][0] += (-cphi[i][1] + term * dix);
      field[i][1] += (-cphi[i][2] + term * diy);
      field[i][2] += (-cphi[i][3] + term * diz);
   }
}

// see also subroutine udirect2b / dfield0c in induce.f
#define DFIELD_DPTRS x, y, z, thole, pdamp, field, fieldp, rpole
void dfieldEwaldReal_acc(real (*field)[3], real (*fieldp)[3])
{
   const real off = switchOff(Switch::EWALD);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DFIELD_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][MPL_PME_0];
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real qixx = rpole[i][MPL_PME_XX];
      real qixy = rpole[i][MPL_PME_XY];
      real qixz = rpole[i][MPL_PME_XZ];
      real qiyy = rpole[i][MPL_PME_YY];
      real qiyz = rpole[i][MPL_PME_YZ];
      real qizz = rpole[i][MPL_PME_ZZ];
      real pdi = pdamp[i];
      real pti = thole[i];
      real gxi = 0, gyi = 0, gzi = 0;
      real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real3 fid = make_real3(0, 0, 0);
            real3 fip = make_real3(0, 0, 0);
            real3 fkd = make_real3(0, 0, 0);
            real3 fkp = make_real3(0, 0, 0);
            pair_dfield<EWALD>(      //
               r2, xr, yr, zr, 1, 1, //
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, pdi,
               pti, //
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z],
               rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ],
               rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], pdamp[k],
               thole[k], //
               aewald, fid, fip, fkd, fkp);

            gxi += fid.x;
            gyi += fid.y;
            gzi += fid.z;
            txi += fip.x;
            tyi += fip.y;
            tzi += fip.z;

            atomic_add(fkd.x, &field[k][0]);
            atomic_add(fkd.y, &field[k][1]);
            atomic_add(fkd.z, &field[k][2]);
            atomic_add(fkp.x, &fieldp[k][0]);
            atomic_add(fkp.y, &fieldp[k][1]);
            atomic_add(fkp.z, &fieldp[k][2]);
         }
      } // end for (int kk)

      atomic_add(gxi, &field[i][0]);
      atomic_add(gyi, &field[i][1]);
      atomic_add(gzi, &field[i][2]);
      atomic_add(txi, &fieldp[i][0]);
      atomic_add(tyi, &fieldp[i][1]);
      atomic_add(tzi, &fieldp[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DFIELD_DPTRS,dpexclude,dpexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < ndpexclude; ++ii) {
      int i = dpexclude[ii][0];
      int k = dpexclude[ii][1];
      real dscale = dpexclude_scale[ii][0] - 1;
      real pscale = dpexclude_scale[ii][1] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][MPL_PME_0];
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real qixx = rpole[i][MPL_PME_XX];
      real qixy = rpole[i][MPL_PME_XY];
      real qixz = rpole[i][MPL_PME_XZ];
      real qiyy = rpole[i][MPL_PME_YY];
      real qiyz = rpole[i][MPL_PME_YZ];
      real qizz = rpole[i][MPL_PME_ZZ];
      real pdi = pdamp[i];
      real pti = thole[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real3 fid = make_real3(0, 0, 0);
         real3 fip = make_real3(0, 0, 0);
         real3 fkd = make_real3(0, 0, 0);
         real3 fkp = make_real3(0, 0, 0);
         pair_dfield<NON_EWALD>(                                             //
            r2, xr, yr, zr, dscale, pscale,                                  //
            ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, pdi, pti, //
            rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y], rpole[k][MPL_PME_Z],
            rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY], rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY],
            rpole[k][MPL_PME_YZ], rpole[k][MPL_PME_ZZ], pdamp[k], thole[k], //
            0, fid, fip, fkd, fkp);

         atomic_add(fid.x, &field[i][0]);
         atomic_add(fid.y, &field[i][1]);
         atomic_add(fid.z, &field[i][2]);
         atomic_add(fip.x, &fieldp[i][0]);
         atomic_add(fip.y, &fieldp[i][1]);
         atomic_add(fip.z, &fieldp[i][2]);

         atomic_add(fkd.x, &field[k][0]);
         atomic_add(fkd.y, &field[k][1]);
         atomic_add(fkd.z, &field[k][2]);
         atomic_add(fkp.x, &fieldp[k][0]);
         atomic_add(fkp.y, &fieldp[k][1]);
         atomic_add(fkp.z, &fieldp[k][2]);
      }
   }
}

// see also subroutine umutual1 in induce.f
void ufieldEwaldRecipSelf_acc(
   const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   darray::zero(g::q0, n, field, fieldp);

   const PMEUnit pu = ppme_unit;
   const auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real aewald = st.aewald;

   cuindToFuind(pu, uind, uinp, fuind, fuinp);
   gridUind(pu, fuind, fuinp);
   fftfront(pu);
   // TODO: store vs. recompute qfac
   pmeConv(pu);
   fftback(pu);
   fphiUind2(pu, fdip_phi1, fdip_phi2);

   const real term = aewald * aewald * aewald * 4 / 3 / sqrtpi;

   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(field,fieldp,uind,uinp,fdip_phi1,fdip_phi2)
   for (int i = 0; i < n; ++i) {
      real a[3][3];
      a[0][0] = nfft1 * recipa.x;
      a[1][0] = nfft2 * recipb.x;
      a[2][0] = nfft3 * recipc.x;
      a[0][1] = nfft1 * recipa.y;
      a[1][1] = nfft2 * recipb.y;
      a[2][1] = nfft3 * recipc.y;
      a[0][2] = nfft1 * recipa.z;
      a[1][2] = nfft2 * recipb.z;
      a[2][2] = nfft3 * recipc.z;

      #pragma acc loop seq
      for (int j = 0; j < 3; ++j) {
         real df1 =
            a[0][j] * fdip_phi1[i][1] + a[1][j] * fdip_phi1[i][2] + a[2][j] * fdip_phi1[i][3];
         real df2 =
            a[0][j] * fdip_phi2[i][1] + a[1][j] * fdip_phi2[i][2] + a[2][j] * fdip_phi2[i][3];
         field[i][j] += (term * uind[i][j] - df1);
         fieldp[i][j] += (term * uinp[i][j] - df2);
      }
   }
}

#define UFIELD_DPTRS x, y, z, thole, pdamp, field, fieldp, uind, uinp
void ufieldEwaldReal_acc(
   const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   const real off = switchOff(Switch::EWALD);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(UFIELD_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uindi0 = uind[i][0];
      real uindi1 = uind[i][1];
      real uindi2 = uind[i][2];
      real uinpi0 = uinp[i][0];
      real uinpi1 = uinp[i][1];
      real uinpi2 = uinp[i][2];
      real pdi = pdamp[i];
      real pti = thole[i];
      real gxi = 0, gyi = 0, gzi = 0;
      real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            real3 fid = make_real3(0, 0, 0);
            real3 fip = make_real3(0, 0, 0);
            real3 fkd = make_real3(0, 0, 0);
            real3 fkp = make_real3(0, 0, 0);
            pair_ufield<EWALD>(                                          //
               r2, xr, yr, zr, 1,                                        //
               uindi0, uindi1, uindi2, uinpi0, uinpi1, uinpi2, pdi, pti, //
               uind[k][0], uind[k][1], uind[k][2], uinp[k][0], uinp[k][1], uinp[k][2], pdamp[k],
               thole[k], //
               aewald, fid, fip, fkd, fkp);

            gxi += fid.x;
            gyi += fid.y;
            gzi += fid.z;
            txi += fip.x;
            tyi += fip.y;
            tzi += fip.z;

            atomic_add(fkd.x, &field[k][0]);
            atomic_add(fkd.y, &field[k][1]);
            atomic_add(fkd.z, &field[k][2]);
            atomic_add(fkp.x, &fieldp[k][0]);
            atomic_add(fkp.y, &fieldp[k][1]);
            atomic_add(fkp.z, &fieldp[k][2]);
         }
      } // end for (int kk)

      atomic_add(gxi, &field[i][0]);
      atomic_add(gyi, &field[i][1]);
      atomic_add(gzi, &field[i][2]);
      atomic_add(txi, &fieldp[i][0]);
      atomic_add(tyi, &fieldp[i][1]);
      atomic_add(tzi, &fieldp[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(UFIELD_DPTRS,uexclude,uexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nuexclude; ++ii) {
      int i = uexclude[ii][0];
      int k = uexclude[ii][1];
      real uscale = uexclude_scale[ii] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uindi0 = uind[i][0];
      real uindi1 = uind[i][1];
      real uindi2 = uind[i][2];
      real uinpi0 = uinp[i][0];
      real uinpi1 = uinp[i][1];
      real uinpi2 = uinp[i][2];
      real pdi = pdamp[i];
      real pti = thole[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2) {
         real3 fid = make_real3(0, 0, 0);
         real3 fip = make_real3(0, 0, 0);
         real3 fkd = make_real3(0, 0, 0);
         real3 fkp = make_real3(0, 0, 0);
         pair_ufield<NON_EWALD>(                                      //
            r2, xr, yr, zr, uscale,                                   //
            uindi0, uindi1, uindi2, uinpi0, uinpi1, uinpi2, pdi, pti, //
            uind[k][0], uind[k][1], uind[k][2], uinp[k][0], uinp[k][1], uinp[k][2], pdamp[k],
            thole[k], //
            0, fid, fip, fkd, fkp);

         atomic_add(fid.x, &field[i][0]);
         atomic_add(fid.y, &field[i][1]);
         atomic_add(fid.z, &field[i][2]);
         atomic_add(fip.x, &fieldp[i][0]);
         atomic_add(fip.y, &fieldp[i][1]);
         atomic_add(fip.z, &fieldp[i][2]);

         atomic_add(fkd.x, &field[k][0]);
         atomic_add(fkd.y, &field[k][1]);
         atomic_add(fkd.z, &field[k][2]);
         atomic_add(fkp.x, &fieldp[k][0]);
         atomic_add(fkp.y, &fieldp[k][1]);
         atomic_add(fkp.z, &fieldp[k][2]);
      }
   }
}
}
