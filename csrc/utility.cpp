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
#include "utility.h"
#include "logging.h"

#define FAIL_IF_ERROR(expr)                       \
  if ((expr) != honey::GetDeviceSuccess()) {        \
    LOG(ERROR) << "Call " << #expr << " failed."; \
    return HoneyError::HoneyFailure;    \
  }

HoneyError HoneyDeviceMalloc(
    void** ptr_out,
    size_t size,
    honey::StreamType stream,
    bool sync) {
  FAIL_IF_ERROR(honey::DeviceMallocAsync(ptr_out, size, stream));
  if (sync) {
    FAIL_IF_ERROR(honey::StreamSynchronize(stream));
  }
  return HoneyError::HoneySuccess;
}

HoneyError HoneyDeviceFree(
    void* ptr,
    honey::StreamType stream,
    bool sync) {
  FAIL_IF_ERROR(honey::FreeDeviceMemoryAsync(ptr, stream));
  if (sync) {
    FAIL_IF_ERROR(honey::StreamSynchronize(stream));
  }
  return HoneyError::HoneySuccess;
}

HoneyError HoneyMemcpy(
    void* dst,
    const void* src,
    size_t count,
    honey::HoneyMemcpyKind kind,
    honey::StreamType stream,
    bool sync) {
  switch (kind) {
    case honey::HoneyMemcpyKind::HostToDevice:
      FAIL_IF_ERROR(honey::CopyToDevice(dst, src, count, stream));
      break;
    case honey::HoneyMemcpyKind::DeviceToHost:
      FAIL_IF_ERROR(honey::CopyToHost(dst, src, count, stream));
      break;
    case honey::HoneyMemcpyKind::DeviceToDevice:
      FAIL_IF_ERROR(honey::DeviceToDeviceCopy(dst, src, count, stream));
      break;
  }
  if (sync) {
    FAIL_IF_ERROR(honey::StreamSynchronize(stream));
  }
  return HoneyError::HoneySuccess;
}
