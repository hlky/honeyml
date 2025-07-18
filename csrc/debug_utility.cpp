//  Copyright 2025 hlky. All rights reserved.

//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#include "debug_utility.h"

namespace {

__global__ void inf_and_nan_checker(const half* tensor, int64_t elem_cnt) {
  int64_t nan_num = 0, pos_inf = 0, neg_inf = 0;
  for (int64_t i = 0; i < elem_cnt; i++) {
    float v = (float)(*(tensor + i));
    if (isnan(v)) {
      nan_num += 1;
    }
    auto is_inf = isinf(v);
    if (is_inf) {
      if (v > 0) {
        pos_inf += 1;
      } else {
        neg_inf += 1;
      }
    }
  }
  if (nan_num > 0 || pos_inf > 0 || neg_inf > 0) {
    printf(
        "contains NaN: %lld, +INF: %lld, -INF: %lld, total elements: %lld\n",
        nan_num,
        pos_inf,
        neg_inf,
        elem_cnt);
  } else {
    printf("doesn't contain NaN/INF\n");
  }
}

} // namespace

namespace honey {
void InvokeInfAndNanChecker(
    const half* tensor,
    const char* tensor_name,
    int64_t elem_cnt,
    honey::StreamType stream) {
  printf("Tensor (%s) ", tensor_name);
  inf_and_nan_checker<<<1, 1, 0, stream>>>(tensor, elem_cnt);
  honey::StreamSynchronize(stream);
}

} // namespace honey
