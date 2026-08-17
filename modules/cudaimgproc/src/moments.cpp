// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.


#include "precomp.hpp"
#include "cuda/moments.cuh"

using namespace cv;
using namespace cv::cuda;

int cv::cuda::numMoments(const MomentsOrder order) {
    return order == MomentsOrder::FIRST_ORDER_MOMENTS ? device::imgproc::n1 : order == MomentsOrder::SECOND_ORDER_MOMENTS ? device::imgproc::n12 : device::imgproc::n123;
}

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
    Moments cv::cuda::moments(InputArray src, const bool binary, const MomentsOrder order, const int momentsType) { throw_no_cuda(); }
    void spatialMoments(InputArray src, OutputArray moments, const bool binary, const MomentsOrder order, const int momentsType, Stream& stream) { throw_no_cuda(); }
#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device { namespace imgproc {
        template <typename TSrc, typename TMoments>
        void moments(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
}}}}

void cv::cuda::spatialMoments(InputArray src, OutputArray moments, const bool binary, const MomentsOrder order, const int momentsType, Stream& stream) {
    CV_Assert(src.depth() <= CV_64F);
    const GpuMat srcDevice = getInputMat(src, stream);

    CV_Assert(momentsType == CV_32F || momentsType == CV_64F);
    const int nMoments = numMoments(order);
    const int momentsCols = nMoments < moments.cols() ? moments.cols() : nMoments;
    GpuMat momentsDevice = getOutputMat(moments, 1, momentsCols, momentsType, stream);
    momentsDevice.setTo(0);

    Point ofs; Size wholeSize;
    srcDevice.locateROI(wholeSize, ofs);

    typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
    static const func_t funcs[7][2] =
    {
        {device::imgproc::moments<uchar, float>,  device::imgproc::moments<uchar, double> },
        {device::imgproc::moments<schar, float>,  device::imgproc::moments<schar, double> },
        {device::imgproc::moments<ushort, float>, device::imgproc::moments<ushort, double>},
        {device::imgproc::moments<short, float>,  device::imgproc::moments<short, double> },
        {device::imgproc::moments<int, float>,    device::imgproc::moments<int, double> },
        {device::imgproc::moments<float, float>,  device::imgproc::moments<float, double> },
        {device::imgproc::moments<double, float>, device::imgproc::moments<double, double> }
    };

    const func_t func = funcs[srcDevice.depth()][momentsType == CV_64F];
    func(srcDevice, momentsDevice, binary, static_cast<int>(order), ofs.x, StreamAccessor::getStream(stream));
    syncOutput(momentsDevice, moments, stream);
}

Moments cv::cuda::moments(InputArray src, const bool binary, const MomentsOrder order, const int momentsType) {
    Stream& stream = Stream::Null();
    HostMem dst;
    spatialMoments(src, dst, binary, order, momentsType, stream);
    stream.waitForCompletion();

    const int nMoments = numMoments(order);
    Mat moments = dst.createMatHeader();
    Mat fullMoments = Mat::zeros(1, 10, CV_64F);

    if (momentsType == CV_32F)
    {
        Mat moments64F;
        moments.convertTo(moments64F, CV_64F);
        moments64F.colRange(0, nMoments).copyTo(fullMoments.colRange(0, nMoments));
    }
    else
    {
        moments.colRange(0, nMoments).copyTo(fullMoments.colRange(0, nMoments));
    }

    return Moments(fullMoments.at<double>(0), fullMoments.at<double>(1), fullMoments.at<double>(2), fullMoments.at<double>(3), fullMoments.at<double>(4), fullMoments.at<double>(5), fullMoments.at<double>(6), fullMoments.at<double>(7), fullMoments.at<double>(8), fullMoments.at<double>(9));
}

#endif /* !defined (HAVE_CUDA) */
