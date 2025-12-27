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
  if ((expr) != dinoml::GetDeviceSuccess()) {        \
    LOG(ERROR) << "Call " << #expr << " failed."; \
    return DinoMLError::DinoMLFailure;    \
  }

DinoMLError DinoMLDeviceMalloc(
    void** ptr_out,
    size_t size,
    dinoml::StreamType stream,
    bool sync) {
  FAIL_IF_ERROR(dinoml::DeviceMallocAsync(ptr_out, size, stream));
  if (sync) {
    FAIL_IF_ERROR(dinoml::StreamSynchronize(stream));
  }
  return DinoMLError::DinoMLSuccess;
}

DinoMLError DinoMLDeviceFree(
    void* ptr,
    dinoml::StreamType stream,
    bool sync) {
  FAIL_IF_ERROR(dinoml::FreeDeviceMemoryAsync(ptr, stream));
  if (sync) {
    FAIL_IF_ERROR(dinoml::StreamSynchronize(stream));
  }
  return DinoMLError::DinoMLSuccess;
}

DinoMLError DinoMLMemcpy(
    void* dst,
    const void* src,
    size_t count,
    dinoml::DinoMLMemcpyKind kind,
    dinoml::StreamType stream,
    bool sync) {
  switch (kind) {
    case dinoml::DinoMLMemcpyKind::HostToDevice:
      FAIL_IF_ERROR(dinoml::CopyToDevice(dst, src, count, stream));
      break;
    case dinoml::DinoMLMemcpyKind::DeviceToHost:
      FAIL_IF_ERROR(dinoml::CopyToHost(dst, src, count, stream));
      break;
    case dinoml::DinoMLMemcpyKind::DeviceToDevice:
      FAIL_IF_ERROR(dinoml::DeviceToDeviceCopy(dst, src, count, stream));
      break;
  }
  if (sync) {
    FAIL_IF_ERROR(dinoml::StreamSynchronize(stream));
  }
  return DinoMLError::DinoMLSuccess;
}
