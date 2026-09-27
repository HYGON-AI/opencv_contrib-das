#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

set +u
source /opt/dtk/env.sh
source /MacCodecSDK/env.sh
for cuda_env in /opt/dtk/cuda/cuda-12/env.sh /opt/dtk/cuda/env.sh /opt/dtk/cuda/cuda/env.sh; do
  if [[ -f "${cuda_env}" ]]; then
    source "${cuda_env}"
  fi
done
set -u

nvcc_path="$(command -v nvcc || true)"
if [[ -z "${nvcc_path}" ]]; then
  nvcc_path="$(find /opt/dtk -type f -path '*/bin/nvcc' -perm /111 -print -quit)"
fi
if [[ -z "${nvcc_path}" ]]; then
  echo "DTK CUDA compiler (nvcc) was not found under /opt/dtk" >&2
  exit 1
fi

export PATH="$(dirname "${nvcc_path}"):${PATH}"
export CUDA_PATH="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
export CUDA_BIN_PATH="${CUDA_PATH}"
export CUDACXX="${nvcc_path}"

"${nvcc_path}" --version
cmake --version

export OPENCV_TEST_DATA_PATH="${GITHUB_WORKSPACE}/opencv_extra/testdata"
export GTEST_FILTER='-CUDA_Event/AsyncEvent.*'

cmake -S opencv -B build-dtk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${nvcc_path}" \
  -DCMAKE_CUDA_COMPILER_FORCED=TRUE \
  -DOPENCV_EXTRA_MODULES_PATH="${GITHUB_WORKSPACE}/modules" \
  -DWITH_CUDA=ON \
  -DBUILD_LIST=alphamat,aruco,bgsegm,bioinspired,calib3d,ccalib,core,cudaarithm,cudabgsegm,cudacodec,cudafeatures2d,cudafilters,cudaimgproc,cudalegacy,cudaobjdetect,cudastereo,cudawarping,cudev,datasets,dnn,dnn_objdetect,dnn_superres,dpm,face,features2d,flann,freetype,fuzzy,gapi,hdf,hfs,highgui,img_hash,imgcodecs,imgproc,intensity_transform,line_descriptor,mcc,ml,objdetect,optflow,phase_unwrapping,photo,plot,python3,quality,rapid,reg,rgbd,saliency,sfm,shape,stereo,stitching,structured_light,superres,surface_matching,text,tracking,ts,video,videostab,viz,wechat_qrcode,xfeatures2d,ximgproc,xobjdetect,xphoto \
  -DBUILD_TESTS=ON \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python3=ON

cmake --build build-dtk --parallel "${MAX_JOBS:-4}"
ctest --test-dir build-dtk --output-on-failure --no-tests=error \
  -R 'opencv_test_(core|cudaarithm)$'
