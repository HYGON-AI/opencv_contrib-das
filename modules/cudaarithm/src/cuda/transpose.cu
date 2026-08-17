/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
// Copyright (C) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

void cv::cuda::transpose(InputArray _src, OutputArray _dst, Stream& stream)
{
    GpuMat src = getInputMat(_src, stream);

    const size_t elemSize = src.elemSize();

    CV_Assert( elemSize == 1 || elemSize == 4 || elemSize == 8 );

    GpuMat dst = getOutputMat(_dst, src.cols, src.rows, src.type(), stream);

    if (elemSize == 1)
    {
        NppStreamHandler h(StreamAccessor::getStream(stream));

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );

        if (!stream)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
    else if (elemSize == 4)
    {
        gridTranspose(globPtr<int>(src), globPtr<int>(dst), stream);
    }
    else // if (elemSize == 8)
    {
        gridTranspose(globPtr<double>(src), globPtr<double>(dst), stream);
    }

    syncOutput(dst, _dst, stream);
}

namespace cv { namespace cuda { namespace device {

namespace
{
    __global__ void flipKernel(const uchar* src, size_t srcStep, uchar* dst, size_t dstStep,
                               int rows, int cols, int elemSize, int flipCode)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= cols || y >= rows)
            return;

        int sx = x;
        int sy = y;
        if (flipCode == 0)
            sy = rows - 1 - y;
        else if (flipCode > 0)
            sx = cols - 1 - x;
        else
        {
            sy = rows - 1 - y;
            sx = cols - 1 - x;
        }

        const uchar* srcPtr = src + sy * srcStep + sx * elemSize;
        uchar* dstPtr = dst + y * dstStep + x * elemSize;
        for (int i = 0; i < elemSize; ++i)
            dstPtr[i] = srcPtr[i];
    }
}

void flip(const GpuMat& src, GpuMat& dst, int flipCode, Stream& stream)
{
    const dim3 block(32, 8);
    const dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    flipKernel<<<grid, block, 0, StreamAccessor::getStream(stream)>>>(
        src.ptr<uchar>(), src.step, dst.ptr<uchar>(), dst.step,
        src.rows, src.cols, static_cast<int>(src.elemSize()), flipCode);

    if (!stream)
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}


namespace
{
__device__ __forceinline__ uchar applyLeftShift(uchar value, int shift)
{
    return static_cast<uchar>(static_cast<unsigned int>(value) << shift);
}

__device__ __forceinline__ schar applyLeftShift(schar value, int shift)
{
    return static_cast<schar>(static_cast<unsigned int>(static_cast<uchar>(value)) << shift);
}

__device__ __forceinline__ ushort applyLeftShift(ushort value, int shift)
{
    return static_cast<ushort>(static_cast<unsigned int>(value) << shift);
}

__device__ __forceinline__ short applyLeftShift(short value, int shift)
{
    return static_cast<short>(static_cast<unsigned int>(static_cast<ushort>(value)) << shift);
}

__device__ __forceinline__ int applyLeftShift(int value, int shift)
{
    return static_cast<int>(static_cast<unsigned int>(value) << shift);
}

__device__ __forceinline__ uchar3 applyLeftShift(uchar3 value, int3 shift)
{
    return make_uchar3(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z));
}

__device__ __forceinline__ char3 applyLeftShift(char3 value, int3 shift)
{
    return make_char3(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z));
}

__device__ __forceinline__ ushort3 applyLeftShift(ushort3 value, int3 shift)
{
    return make_ushort3(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z));
}

__device__ __forceinline__ short3 applyLeftShift(short3 value, int3 shift)
{
    return make_short3(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z));
}

__device__ __forceinline__ int3 applyLeftShift(int3 value, int3 shift)
{
    return make_int3(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z));
}

__device__ __forceinline__ uchar4 applyLeftShift(uchar4 value, int4 shift)
{
    return make_uchar4(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z), applyLeftShift(value.w, shift.w));
}

__device__ __forceinline__ char4 applyLeftShift(char4 value, int4 shift)
{
    return make_char4(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z), applyLeftShift(value.w, shift.w));
}

__device__ __forceinline__ ushort4 applyLeftShift(ushort4 value, int4 shift)
{
    return make_ushort4(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z), applyLeftShift(value.w, shift.w));
}

__device__ __forceinline__ short4 applyLeftShift(short4 value, int4 shift)
{
    return make_short4(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z), applyLeftShift(value.w, shift.w));
}

__device__ __forceinline__ int4 applyLeftShift(int4 value, int4 shift)
{
    return make_int4(applyLeftShift(value.x, shift.x), applyLeftShift(value.y, shift.y), applyLeftShift(value.z, shift.z), applyLeftShift(value.w, shift.w));
}

__device__ __forceinline__ uchar applyRightShift(uchar value, int shift)
{
    return static_cast<uchar>(value >> shift);
}

__device__ __forceinline__ schar applyRightShift(schar value, int shift)
{
    return static_cast<schar>(value >> shift);
}

__device__ __forceinline__ ushort applyRightShift(ushort value, int shift)
{
    return static_cast<ushort>(value >> shift);
}

__device__ __forceinline__ short applyRightShift(short value, int shift)
{
    return static_cast<short>(value >> shift);
}

__device__ __forceinline__ int applyRightShift(int value, int shift)
{
    return value >> shift;
}

__device__ __forceinline__ uchar3 applyRightShift(uchar3 value, int3 shift)
{
    return make_uchar3(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z));
}

__device__ __forceinline__ char3 applyRightShift(char3 value, int3 shift)
{
    return make_char3(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z));
}

__device__ __forceinline__ ushort3 applyRightShift(ushort3 value, int3 shift)
{
    return make_ushort3(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z));
}

__device__ __forceinline__ short3 applyRightShift(short3 value, int3 shift)
{
    return make_short3(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z));
}

__device__ __forceinline__ int3 applyRightShift(int3 value, int3 shift)
{
    return make_int3(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z));
}

__device__ __forceinline__ uchar4 applyRightShift(uchar4 value, int4 shift)
{
    return make_uchar4(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z), applyRightShift(value.w, shift.w));
}

__device__ __forceinline__ char4 applyRightShift(char4 value, int4 shift)
{
    return make_char4(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z), applyRightShift(value.w, shift.w));
}

__device__ __forceinline__ ushort4 applyRightShift(ushort4 value, int4 shift)
{
    return make_ushort4(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z), applyRightShift(value.w, shift.w));
}

__device__ __forceinline__ short4 applyRightShift(short4 value, int4 shift)
{
    return make_short4(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z), applyRightShift(value.w, shift.w));
}

__device__ __forceinline__ int4 applyRightShift(int4 value, int4 shift)
{
    return make_int4(applyRightShift(value.x, shift.x), applyRightShift(value.y, shift.y), applyRightShift(value.z, shift.z), applyRightShift(value.w, shift.w));
}

template <typename T, typename ShiftType>
__global__ void lshiftKernel(const uchar* src, size_t srcStep, uchar* dst, size_t dstStep,
                             int rows, int cols, ShiftType shift)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;

    const T* srcRow = reinterpret_cast<const T*>(src + y * srcStep);
    T* dstRow = reinterpret_cast<T*>(dst + y * dstStep);
    dstRow[x] = applyLeftShift(srcRow[x], shift);
}

template <typename T, typename ShiftType>
__global__ void rshiftKernel(const uchar* src, size_t srcStep, uchar* dst, size_t dstStep,
                             int rows, int cols, ShiftType shift)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows)
        return;

    const T* srcRow = reinterpret_cast<const T*>(src + y * srcStep);
    T* dstRow = reinterpret_cast<T*>(dst + y * dstStep);
    dstRow[x] = applyRightShift(srcRow[x], shift);
}

template <typename T> struct ShiftValue;
template <> struct ShiftValue<uchar>   { typedef int type;  static type make(Scalar_<int> v) { return v[0]; } };
template <> struct ShiftValue<schar>   { typedef int type;  static type make(Scalar_<int> v) { return v[0]; } };
template <> struct ShiftValue<ushort>  { typedef int type;  static type make(Scalar_<int> v) { return v[0]; } };
template <> struct ShiftValue<short>   { typedef int type;  static type make(Scalar_<int> v) { return v[0]; } };
template <> struct ShiftValue<int>     { typedef int type;  static type make(Scalar_<int> v) { return v[0]; } };
template <> struct ShiftValue<uchar3>  { typedef int3 type; static type make(Scalar_<int> v) { return make_int3(v[0], v[1], v[2]); } };
template <> struct ShiftValue<char3>   { typedef int3 type; static type make(Scalar_<int> v) { return make_int3(v[0], v[1], v[2]); } };
template <> struct ShiftValue<ushort3> { typedef int3 type; static type make(Scalar_<int> v) { return make_int3(v[0], v[1], v[2]); } };
template <> struct ShiftValue<short3>  { typedef int3 type; static type make(Scalar_<int> v) { return make_int3(v[0], v[1], v[2]); } };
template <> struct ShiftValue<int3>    { typedef int3 type; static type make(Scalar_<int> v) { return make_int3(v[0], v[1], v[2]); } };
template <> struct ShiftValue<uchar4>  { typedef int4 type; static type make(Scalar_<int> v) { return make_int4(v[0], v[1], v[2], v[3]); } };
template <> struct ShiftValue<char4>   { typedef int4 type; static type make(Scalar_<int> v) { return make_int4(v[0], v[1], v[2], v[3]); } };
template <> struct ShiftValue<ushort4> { typedef int4 type; static type make(Scalar_<int> v) { return make_int4(v[0], v[1], v[2], v[3]); } };
template <> struct ShiftValue<short4>  { typedef int4 type; static type make(Scalar_<int> v) { return make_int4(v[0], v[1], v[2], v[3]); } };
template <> struct ShiftValue<int4>    { typedef int4 type; static type make(Scalar_<int> v) { return make_int4(v[0], v[1], v[2], v[3]); } };

template <typename T>
void launchLShiftImpl(const GpuMat& src, Scalar_<int> val, GpuMat& dst, Stream& stream)
{
    typedef typename ShiftValue<T>::type ShiftType;
    const dim3 block(32, 8);
    const dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    lshiftKernel<T, ShiftType><<<grid, block, 0, StreamAccessor::getStream(stream)>>>(
        src.ptr<uchar>(), src.step, dst.ptr<uchar>(), dst.step, src.rows, src.cols, ShiftValue<T>::make(val));

    if (!stream)
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

template <typename T>
void launchRShiftImpl(const GpuMat& src, Scalar_<int> val, GpuMat& dst, Stream& stream)
{
    typedef typename ShiftValue<T>::type ShiftType;
    const dim3 block(32, 8);
    const dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    rshiftKernel<T, ShiftType><<<grid, block, 0, StreamAccessor::getStream(stream)>>>(
        src.ptr<uchar>(), src.step, dst.ptr<uchar>(), dst.step, src.rows, src.cols, ShiftValue<T>::make(val));

    if (!stream)
        CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}
}

void rshift(const GpuMat& src, Scalar_<int> val, GpuMat& dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat&, Scalar_<int>, GpuMat&, Stream&);
    static const func_t funcs[5][4] =
    {
        {launchRShiftImpl<uchar>, 0, launchRShiftImpl<uchar3>, launchRShiftImpl<uchar4>},
        {launchRShiftImpl<schar>, 0, launchRShiftImpl<char3>, launchRShiftImpl<char4>},
        {launchRShiftImpl<ushort>, 0, launchRShiftImpl<ushort3>, launchRShiftImpl<ushort4>},
        {launchRShiftImpl<short>, 0, launchRShiftImpl<short3>, launchRShiftImpl<short4>},
        {launchRShiftImpl<int>, 0, launchRShiftImpl<int3>, launchRShiftImpl<int4>}
    };

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func);
    func(src, val, dst, stream);
}

void lshift(const GpuMat& src, Scalar_<int> val, GpuMat& dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat&, Scalar_<int>, GpuMat&, Stream&);
    static const func_t funcs[5][4] =
    {
        {launchLShiftImpl<uchar>, 0, launchLShiftImpl<uchar3>, launchLShiftImpl<uchar4>},
        {0, 0, 0, 0},
        {launchLShiftImpl<ushort>, 0, launchLShiftImpl<ushort3>, launchLShiftImpl<ushort4>},
        {0, 0, 0, 0},
        {launchLShiftImpl<int>, 0, launchLShiftImpl<int3>, launchLShiftImpl<int4>}
    };

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func);
    func(src, val, dst, stream);
}

}}}

#endif
