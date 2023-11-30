#!/bin/bash
#==========================================
# Copyright © Intel Corporation
#
# SPDX-License-Identifier: MIT
#==========================================
# Script to submit job in Intel(R) DevCloud
# Version: 0.71
#==========================================
#启用oneAPI环境
. /opt/intel/oneapi/setvars.sh  > /dev/null
if [ -z "$(icpx --version)" ]; then
  echo "oneAPI env has some problem! Please check your settings." 
  echo "oneAPI env has some problem! Please check your settings."
  echo "oneAPI env has some problem! Please check your settings."
else
  echo "oneAPI env is OK!"
  echo "oneAPI env is OK!"
  echo "oneAPI env is OK!"
fi

#检查sycl环境是否配置成功
sycl-ls

# 先编译
icpx -fsycl test.cpp
#运行
./a.out 10
