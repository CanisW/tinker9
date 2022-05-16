#pragma once
#include "tool/externfunc.h"
#include <cstddef>

// host
namespace tinker {
/// \ingroup math
/// \brief Zeros variable on host.
inline void zeroOnHost(int& v)
{
   v = 0;
}

/// \ingroup math
/// \brief Zeros variable on host.
inline void zeroOnHost(float& v)
{
   v = 0;
}

/// \ingroup math
/// \brief Zeros variable on host.
inline void zeroOnHost(double& v)
{
   v = 0;
}

/// \ingroup math
/// \brief Zeros variable on host.
inline void zeroOnHost(unsigned long long& v)
{
   v = 0;
}

/// \ingroup math
/// \brief Zeros variable on host.
template <class T>
void zeroOnHost(T*& ptr)
{
   ptr = nullptr;
}

/// \ingroup math
/// \brief Zeros variables on host.
template <class T, size_t N>
void zeroOnHost(T (&v)[N])
{
   for (size_t i = 0; i < N; ++i)
      zeroOnHost(v[i]);
}

/// \ingroup math
/// \brief Zeros variables on host.
template <class T, class... Ts>
void zeroOnHost(T& v, Ts&... vs)
{
   zeroOnHost(v);
   zeroOnHost(vs...);
}
}

// device
namespace tinker {
template <class T>
void zeroOnDevice3Async_acc(int nelem, T* a1, T* a2, T* a3);
template <class T>
void zeroOnDevice3Async_cu(int nelem, T* a1, T* a2, T* a3);

/// \ingroup math
/// \brief Zeros variables on device.
template <class T>
void zeroOnDevice3Async(int nelem, T* a1, T* a2, T* a3)
{
   // TINKER_FVOID2(acc1, cu1, zeroOnDevice3Async, int, T*, T*, T*);
   TINKER_FCALL2(acc1, cu1, zeroOnDevice3Async, nelem, a1, a2, a3);
}

template <class T, int N>
void zeroOnDevice3Async_acc(int nelem, T (*a1)[N], T (*a2)[N], T (*a3)[N]);
template <class T, int N>
void zeroOnDevice3Async_cu(int nelem, T (*a1)[N], T (*a2)[N], T (*a3)[N]);

/// \ingroup math
/// \brief Zeros variables on device.
template <class T, int N>
void zeroOnDevice3Async(int nelem, T (*a1)[N], T (*a2)[N], T (*a3)[N])
{
   // TINKER_FVOID2(acc1, cu1, zeroOnDevice3Async, int, T (*)[N], T (*)[N], T (*)[N]);
   TINKER_FCALL2(acc1, cu1, zeroOnDevice3Async, nelem, a1, a2, a3);
}

template <class T>
void zeroOnDevice9Async_acc(
   int nelem, T* a1, T* a2, T* a3, T* a4, T* a5, T* a6, T* a7, T* a8, T* a9);
template <class T>
void zeroOnDevice9Async_cu(
   int nelem, T* a1, T* a2, T* a3, T* a4, T* a5, T* a6, T* a7, T* a8, T* a9);

/// \ingroup math
/// \brief Zeros variables on device.
template <class T>
void zeroOnDevice9Async(int nelem, T* a1, T* a2, T* a3, T* a4, T* a5, T* a6, T* a7, T* a8, T* a9)
{
   // TINKER_FVOID2(acc1, cu1, zeroOnDevice9Async, ...);
   TINKER_FCALL2(acc1, cu1, zeroOnDevice9Async, nelem, a1, a2, a3, a4, a5, a6, a7, a8, a9);
}
}
