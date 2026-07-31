/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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

#pragma once

#ifndef OPENCV_CUDEV_BLOCK_REDUCE_HPP
#define OPENCV_CUDEV_BLOCK_REDUCE_HPP

#include "../common.hpp"
#include "../util/tuple.hpp"
#include "../warp/reduce.hpp"
#include "detail/reduce.hpp"
#include "detail/reduce_key_val.hpp"
#include <type_traits>

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// blockReduce

#ifdef DAS_ADAPT
namespace block_reduce_compat
{
    template <int N, class SmemTuple, class ValTuple, class OpTuple, int SIZE>
    struct TupleReduce;

    template <int N, class SmemTuple, class ValTuple, class OpTuple>
    struct TupleReduce<N, SmemTuple, ValTuple, OpTuple, 2>
    {
        static __device__ __forceinline__ void run(const SmemTuple& smem, const ValTuple& val, uint tid, const OpTuple& op)
        {
            SmemTuple sm = smem;
            ValTuple vl = val;
            OpTuple ops = op;
            blockReduce<N>(get<0>(sm), get<0>(vl), tid, get<0>(ops));
            blockReduce<N>(get<1>(sm), get<1>(vl), tid, get<1>(ops));
        }
    };

    template <int N, class SmemTuple, class ValTuple, class OpTuple>
    struct TupleReduce<N, SmemTuple, ValTuple, OpTuple, 3>
    {
        static __device__ __forceinline__ void run(const SmemTuple& smem, const ValTuple& val, uint tid, const OpTuple& op)
        {
            SmemTuple sm = smem;
            ValTuple vl = val;
            OpTuple ops = op;
            blockReduce<N>(get<0>(sm), get<0>(vl), tid, get<0>(ops));
            blockReduce<N>(get<1>(sm), get<1>(vl), tid, get<1>(ops));
            blockReduce<N>(get<2>(sm), get<2>(vl), tid, get<2>(ops));
        }
    };

    template <int N, class SmemTuple, class ValTuple, class OpTuple>
    struct TupleReduce<N, SmemTuple, ValTuple, OpTuple, 4>
    {
        static __device__ __forceinline__ void run(const SmemTuple& smem, const ValTuple& val, uint tid, const OpTuple& op)
        {
            SmemTuple sm = smem;
            ValTuple vl = val;
            OpTuple ops = op;
            blockReduce<N>(get<0>(sm), get<0>(vl), tid, get<0>(ops));
            blockReduce<N>(get<1>(sm), get<1>(vl), tid, get<1>(ops));
            blockReduce<N>(get<2>(sm), get<2>(vl), tid, get<2>(ops));
            blockReduce<N>(get<3>(sm), get<3>(vl), tid, get<3>(ops));
        }
    };

    template <int N, class KeyTuple, class KeyValTuple, class ValTuple, class ValValTuple, class CmpTuple, int SIZE>
    struct TupleReduceKeyVal;

    template <int N, class KeyTuple, class KeyValTuple, class ValTuple, class ValValTuple, class CmpTuple>
    struct TupleReduceKeyVal<N, KeyTuple, KeyValTuple, ValTuple, ValValTuple, CmpTuple, 2>
    {
        static __device__ __forceinline__ void run(const KeyTuple& skeys, const KeyValTuple& key, const ValTuple& svals, const ValValTuple& val, uint tid, const CmpTuple& cmp)
        {
            KeyTuple ks = skeys;
            KeyValTuple kv = key;
            ValTuple vs = svals;
            ValValTuple vv = val;
            CmpTuple cp = cmp;
            blockReduceKeyVal<N>(get<0>(ks), get<0>(kv), get<0>(vs), get<0>(vv), tid, get<0>(cp));
            blockReduceKeyVal<N>(get<1>(ks), get<1>(kv), get<1>(vs), get<1>(vv), tid, get<1>(cp));
        }
    };
}
#endif

template <int N, typename T, class Op>
__device__ __forceinline__ void blockReduce(volatile T* smem, T& val, uint tid, const Op& op)
{
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<volatile T*, T&, const Op&>(smem, val, tid, op);
}

template <int N, typename P0, typename P1, typename R0, typename R1, class Op0, class Op1>
__device__ __forceinline__ void blockReduce(const tuple<P0, P1>& smem,
                                            const tuple<R0, R1>& val,
                                            uint tid,
                                            const tuple<Op0, Op1>& op)
{
#ifdef DAS_ADAPT
    blockReduce<N>(get<0>(smem), get<0>(val), tid, get<0>(op));
    blockReduce<N>(get<1>(smem), get<1>(val), tid, get<1>(op));
#else
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<P0, P1>&,
            const tuple<R0, R1>&,
            const tuple<Op0, Op1>&>(smem, val, tid, op);
#endif
}

template <int N, typename P0, typename P1, typename P2, typename R0, typename R1, typename R2, class Op0, class Op1, class Op2>
__device__ __forceinline__ void blockReduce(const tuple<P0, P1, P2>& smem,
                                            const tuple<R0, R1, R2>& val,
                                            uint tid,
                                            const tuple<Op0, Op1, Op2>& op)
{
#ifdef DAS_ADAPT
    blockReduce<N>(get<0>(smem), get<0>(val), tid, get<0>(op));
    blockReduce<N>(get<1>(smem), get<1>(val), tid, get<1>(op));
    blockReduce<N>(get<2>(smem), get<2>(val), tid, get<2>(op));
#else
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<P0, P1, P2>&,
            const tuple<R0, R1, R2>&,
            const tuple<Op0, Op1, Op2>&>(smem, val, tid, op);
#endif
}

template <int N, typename P0, typename P1, typename P2, typename P3, typename R0, typename R1, typename R2, typename R3, class Op0, class Op1, class Op2, class Op3>
__device__ __forceinline__ void blockReduce(const tuple<P0, P1, P2, P3>& smem,
                                            const tuple<R0, R1, R2, R3>& val,
                                            uint tid,
                                            const tuple<Op0, Op1, Op2, Op3>& op)
{
#ifdef DAS_ADAPT
    blockReduce<N>(get<0>(smem), get<0>(val), tid, get<0>(op));
    blockReduce<N>(get<1>(smem), get<1>(val), tid, get<1>(op));
    blockReduce<N>(get<2>(smem), get<2>(val), tid, get<2>(op));
    blockReduce<N>(get<3>(smem), get<3>(val), tid, get<3>(op));
#else
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<P0, P1, P2, P3>&,
            const tuple<R0, R1, R2, R3>&,
            const tuple<Op0, Op1, Op2, Op3>&>(smem, val, tid, op);
#endif
}

template <int N,
          typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
          typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
          class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
__device__ __forceinline__ void blockReduce(const tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                            const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                            uint tid,
                                            const tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
{
#ifndef DAS_ADAPT
    block_reduce_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>&,
            const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>&,
            const tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>&>(smem, val, tid, op);
#else
    block_reduce_compat::TupleReduce<N,
            tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>,
            tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>,
            tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>,
            tuple_size< tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value>::run(smem, val, tid, op);
#endif
}

// blockReduceKeyVal

template <int N, typename K, typename V, class Cmp>
__device__ __forceinline__ void blockReduceKeyVal(volatile K* skeys, K& key, volatile V* svals, V& val, uint tid, const Cmp& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&, volatile V*, V&, const Cmp&>(skeys, key, svals, val, tid, cmp);
}

template <int N,
          typename K,
          typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
          typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
          class Cmp>
__device__ __forceinline__ void blockReduceKeyVal(volatile K* skeys, K& key,
                                                  const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                                  const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                                  uint tid, const Cmp& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&,
            const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>&,
            const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>&,
            const Cmp&>(skeys, key, svals, val, tid, cmp);
}

template <int N,
          typename KP0, typename KP1,
          typename KR0, typename KR1,
          typename VP0, typename VP1,
          typename VR0, typename VR1,
          class Cmp0, class Cmp1>
__device__ __forceinline__ void blockReduceKeyVal(const tuple<KP0, KP1>& skeys,
                                                  const tuple<KR0, KR1>& key,
                                                  const tuple<VP0, VP1>& svals,
                                                  const tuple<VR0, VR1>& val,
                                                  uint tid,
                                                  const tuple<Cmp0, Cmp1>& cmp)
{
#ifdef DAS_ADAPT
    typedef typename std::remove_cv<typename std::remove_pointer<VP0>::type>::type V0;
    typedef typename std::remove_cv<typename std::remove_pointer<VP1>::type>::type V1;
    typedef typename std::remove_cv<typename std::remove_reference<VR0>::type>::type R0;
    typedef typename std::remove_cv<typename std::remove_reference<VR1>::type>::type R1;

    V0 val0 = static_cast<V0>(get<0>(val));
    V1 val1 = static_cast<V1>(get<1>(val));

    blockReduceKeyVal<N>(get<0>(skeys), get<0>(key), get<0>(svals), val0, tid, get<0>(cmp));
    blockReduceKeyVal<N>(get<1>(skeys), get<1>(key), get<1>(svals), val1, tid, get<1>(cmp));

    get<0>(val) = static_cast<R0>(val0);
    get<1>(val) = static_cast<R1>(val1);
#else
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<KP0, KP1>&,
            const tuple<KR0, KR1>&,
            const tuple<VP0, VP1>&,
            const tuple<VR0, VR1>&,
            const tuple<Cmp0, Cmp1>&>(skeys, key, svals, val, tid, cmp);
#endif
}

template <int N,
          typename KP0, typename KP1, typename KP2, typename KP3, typename KP4, typename KP5, typename KP6, typename KP7, typename KP8, typename KP9,
          typename KR0, typename KR1, typename KR2, typename KR3, typename KR4, typename KR5, typename KR6, typename KR7, typename KR8, typename KR9,
          typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
          typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
          class Cmp0, class Cmp1, class Cmp2, class Cmp3, class Cmp4, class Cmp5, class Cmp6, class Cmp7, class Cmp8, class Cmp9>
__device__ __forceinline__ void blockReduceKeyVal(const tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>& skeys,
                                                  const tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>& key,
                                                  const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                                  const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                                  uint tid,
                                                  const tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>& cmp)
{
    block_reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<
            const tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>&,
            const tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>&,
            const tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>&,
            const tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>&,
            const tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>&
            >(skeys, key, svals, val, tid, cmp);
}

//! @}

}}

#endif
