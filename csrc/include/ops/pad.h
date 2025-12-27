#pragma once
#include "dinoml/device.h"

namespace dinoml {

template <typename T, typename IndexType>
__global__ void pad_kernel_constant_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType pad_left,
    IndexType pad_right,
    T pad_value) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N + pad_left + pad_right;
  if (idx < total_elements) {
    if (idx < pad_left || idx >= N + pad_left) {
      output[idx] = pad_value;
    } else {
      output[idx] = input[idx - pad_left];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_reflect_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N + pad_left + pad_right;
  if (idx < total_elements) {
    if (idx < pad_left) {
      output[idx] = input[pad_left - idx - 1];
    } else if (idx >= N + pad_left) {
      output[idx] = input[2 * N + pad_left - idx - 1];
    } else {
      output[idx] = input[idx - pad_left];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_replicate_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N + pad_left + pad_right;
  if (idx < total_elements) {
    if (idx < pad_left) {
      output[idx] = input[0];
    } else if (idx >= N + pad_left) {
      output[idx] = input[N - 1];
    } else {
      output[idx] = input[idx - pad_left];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_circular_1d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N + pad_left + pad_right;
  if (idx < total_elements) {
    if (idx < pad_left) {
      output[idx] = input[(idx - pad_left + N) % N];
    } else if (idx >= N + pad_left) {
      output[idx] = input[(idx - pad_left) % N];
    } else {
      output[idx] = input[idx - pad_left];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_constant_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType pad_left,
    IndexType pad_right,
    T pad_value) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N * (H + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType h = idx % (H + pad_left + pad_right);
    IndexType n = idx / (H + pad_left + pad_right);

    if (h < pad_left || h >= H + pad_left) {
      output[idx] = pad_value;
    } else {
      IndexType in_h = h - pad_left;
      IndexType in_index = n * H + in_h;
      output[idx] = input[in_index];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_reflect_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N * (H + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType h = idx % (H + pad_left + pad_right);
    IndexType n = idx / (H + pad_left + pad_right);

    IndexType in_h = h < pad_left
        ? pad_left - h
        : (h >= H + pad_left ? 2 * H + pad_left - h - 2 : h - pad_left);
    IndexType in_index = n * H + in_h;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_replicate_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N * (H + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType h = idx % (H + pad_left + pad_right);
    IndexType n = idx / (H + pad_left + pad_right);

    IndexType in_h =
        std::min<IndexType>(std::max<IndexType>(h - pad_left, 0), H - 1);
    IndexType in_index = n * H + in_h;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_circular_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements = N * (H + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType h = idx % (H + pad_left + pad_right);
    IndexType n = idx / (H + pad_left + pad_right);

    IndexType in_h = (h - pad_left + H) % H;
    IndexType in_index = n * H + in_h;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_constant_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right,
    T pad_value) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType w = idx % (W + pad_left + pad_right);
    IndexType h =
        (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    if (h < pad_top || h >= H + pad_top || w < pad_left || w >= W + pad_left) {
      output[idx] = pad_value;
    } else {
      IndexType in_h = h - pad_top;
      IndexType in_w = w - pad_left;
      IndexType in_index = n * H * W + in_h * W + in_w;
      output[idx] = input[in_index];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_reflect_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType w = idx % (W + pad_left + pad_right);
    IndexType h =
        (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    IndexType in_h = h < pad_top
        ? pad_top - h
        : (h >= H + pad_top ? 2 * H + pad_top - h - 2 : h - pad_top);
    IndexType in_w = w < pad_left
        ? pad_left - w
        : (w >= W + pad_left ? 2 * W + pad_left - w - 2 : w - pad_left);
    IndexType in_index = n * H * W + in_h * W + in_w;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_replicate_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType w = idx % (W + pad_left + pad_right);
    IndexType h =
        (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    IndexType in_h =
        std::min<IndexType>(std::max<IndexType>(h - pad_top, 0), H - 1);
    IndexType in_w =
        std::min<IndexType>(std::max<IndexType>(w - pad_left, 0), W - 1);
    IndexType in_index = n * H * W + in_h * W + in_w;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_circular_3d(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
  if (idx < total_elements) {
    IndexType w = idx % (W + pad_left + pad_right);
    IndexType h =
        (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / ((H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    IndexType in_h = (h - pad_top + H) % H;
    IndexType in_w = (w - pad_left + W) % W;
    IndexType in_index = n * H * W + in_h * W + in_w;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_constant_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right,
    T pad_value) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    if (h < pad_top || h >= H + pad_top || w < pad_left || w >= W + pad_left) {
      output[idx] = pad_value;
    } else {
      IndexType in_h = h - pad_top;
      IndexType in_w = w - pad_left;
      IndexType in_index = n * H * W * C + in_h * W * C + in_w * C + c;
      output[idx] = input[in_index];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_reflect_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    IndexType in_h = h < pad_top
        ? pad_top - h
        : (h >= H + pad_top ? 2 * H + pad_top - h - 2 : h - pad_top);
    IndexType in_w = w < pad_left
        ? pad_left - w
        : (w >= W + pad_left ? 2 * W + pad_left - w - 2 : w - pad_left);
    IndexType in_index = n * H * W * C + in_h * W * C + in_w * C + c;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_replicate_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    IndexType in_h =
        std::min<IndexType>(std::max<IndexType>(h - pad_top, 0), H - 1);
    IndexType in_w =
        std::min<IndexType>(std::max<IndexType>(w - pad_left, 0), W - 1);
    IndexType in_index = n * H * W * C + in_h * W * C + in_w * C + c;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_circular_4d_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType total_elements =
      N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType n =
        idx / (C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right));

    IndexType in_h = (h - pad_top + H) % H;
    IndexType in_w = (w - pad_left + W) % W;
    IndexType in_index = n * H * W * C + in_h * W * C + in_w * C + c;
    output[idx] = input[in_index];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_constant_5d_ndhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType D,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_front,
    IndexType pad_back,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right,
    T pad_value) {
  IndexType total_elements = N * (D + pad_front + pad_back) *
      (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType d =
        (idx / (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom))) %
        (D + pad_front + pad_back);
    IndexType n = idx /
        (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom) *
         (D + pad_front + pad_back));

    if (d < pad_front || d >= (D + pad_front) || h < pad_top ||
        h >= (H + pad_top) || w < pad_left || w >= (W + pad_left)) {
      output[idx] = pad_value;
    } else {
      IndexType in_d = d - pad_front;
      IndexType in_h = h - pad_top;
      IndexType in_w = w - pad_left;
      IndexType in_idx = n * (D * H * W * C) + in_d * (H * W * C) +
          in_h * (W * C) + in_w * C + c;
      output[idx] = input[in_idx];
    }
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_reflect_5d_ndhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType D,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_front,
    IndexType pad_back,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType total_elements = N * (D + pad_front + pad_back) *
      (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType d =
        (idx / (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom))) %
        (D + pad_front + pad_back);
    IndexType n = idx /
        (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom) *
         (D + pad_front + pad_back));

    IndexType in_d = (d < pad_front)
        ? (pad_front - d)
        : (d >= (D + pad_front) ? 2 * D + pad_front - d - 2 : d - pad_front);
    IndexType in_h = (h < pad_top)
        ? (pad_top - h)
        : (h >= (H + pad_top) ? 2 * H + pad_top - h - 2 : h - pad_top);
    IndexType in_w = (w < pad_left)
        ? (pad_left - w)
        : (w >= (W + pad_left) ? 2 * W + pad_left - w - 2 : w - pad_left);

    IndexType in_idx = n * (D * H * W * C) + in_d * (H * W * C) +
        in_h * (W * C) + in_w * C + c;
    output[idx] = input[in_idx];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_replicate_5d_ndhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType D,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_front,
    IndexType pad_back,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType total_elements = N * (D + pad_front + pad_back) *
      (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType d =
        (idx / (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom))) %
        (D + pad_front + pad_back);
    IndexType n = idx /
        (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom) *
         (D + pad_front + pad_back));

    IndexType in_d = min(max(d - pad_front, (IndexType)0), D - 1);
    IndexType in_h = min(max(h - pad_top, (IndexType)0), H - 1);
    IndexType in_w = min(max(w - pad_left, (IndexType)0), W - 1);

    IndexType in_idx = n * (D * H * W * C) + in_d * (H * W * C) +
        in_h * (W * C) + in_w * C + c;
    output[idx] = input[in_idx];
  }
}

template <typename T, typename IndexType>
__global__ void pad_kernel_circular_5d_ndhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    IndexType N,
    IndexType D,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_front,
    IndexType pad_back,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right) {
  IndexType total_elements = N * (D + pad_front + pad_back) *
      (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    IndexType c = idx % C;
    IndexType w = (idx / C) % (W + pad_left + pad_right);
    IndexType h =
        (idx / (C * (W + pad_left + pad_right))) % (H + pad_top + pad_bottom);
    IndexType d =
        (idx / (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom))) %
        (D + pad_front + pad_back);
    IndexType n = idx /
        (C * (W + pad_left + pad_right) * (H + pad_top + pad_bottom) *
         (D + pad_front + pad_back));

    IndexType in_d = (d - pad_front + D) % D;
    IndexType in_h = (h - pad_top + H) % H;
    IndexType in_w = (w - pad_left + W) % W;

    IndexType in_idx = n * (D * H * W * C) + in_d * (H * W * C) +
        in_h * (W * C) + in_w * C + c;
    output[idx] = input[in_idx];
  }
}

} // namespace dinoml

template <typename IndexType, typename ElemInputType, typename ElemOutputType>
void InvokePad(
    const void* in_ptr,
    void* out_ptr,
    IndexType N,
    IndexType D,
    IndexType H,
    IndexType W,
    IndexType C,
    IndexType pad_front,
    IndexType pad_back,
    IndexType pad_top,
    IndexType pad_bottom,
    IndexType pad_left,
    IndexType pad_right,
    ElemInputType pad_value,
    IndexType rank,
    const char* mode,
    dinoml::DeviceStream stream) {
  IndexType total_elements;
  IndexType threads_per_block = 256;
  IndexType num_blocks;

  if (rank == 1) {
    total_elements = N + pad_left + pad_right;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (strcmp(mode, "constant") == 0) {
      dinoml::pad_kernel_constant_1d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          pad_left,
          pad_right,
          pad_value);
    } else if (strcmp(mode, "reflect") == 0) {
      dinoml::pad_kernel_reflect_1d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "replicate") == 0) {
      dinoml::
          pad_kernel_replicate_1d<<<num_blocks, threads_per_block, 0, stream>>>(
              static_cast<const ElemInputType*>(in_ptr),
              static_cast<ElemOutputType*>(out_ptr),
              N,
              pad_left,
              pad_right);
    } else if (strcmp(mode, "circular") == 0) {
      dinoml::pad_kernel_circular_1d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          pad_left,
          pad_right);
    }
  } else if (rank == 2) {
    total_elements = N * (H + pad_left + pad_right);
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (strcmp(mode, "constant") == 0) {
      dinoml::pad_kernel_constant_2d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          pad_left,
          pad_right,
          pad_value);
    } else if (strcmp(mode, "reflect") == 0) {
      dinoml::pad_kernel_reflect_2d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "replicate") == 0) {
      dinoml::
          pad_kernel_replicate_2d<<<num_blocks, threads_per_block, 0, stream>>>(
              static_cast<const ElemInputType*>(in_ptr),
              static_cast<ElemOutputType*>(out_ptr),
              N,
              H,
              pad_left,
              pad_right);
    } else if (strcmp(mode, "circular") == 0) {
      dinoml::pad_kernel_circular_2d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          pad_left,
          pad_right);
    }
  } else if (rank == 3) {
    total_elements =
        N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (strcmp(mode, "constant") == 0) {
      dinoml::pad_kernel_constant_3d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right,
          pad_value);
    } else if (strcmp(mode, "reflect") == 0) {
      dinoml::pad_kernel_reflect_3d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "replicate") == 0) {
      dinoml::
          pad_kernel_replicate_3d<<<num_blocks, threads_per_block, 0, stream>>>(
              static_cast<const ElemInputType*>(in_ptr),
              static_cast<ElemOutputType*>(out_ptr),
              N,
              H,
              W,
              pad_top,
              pad_bottom,
              pad_left,
              pad_right);
    } else if (strcmp(mode, "circular") == 0) {
      dinoml::pad_kernel_circular_3d<<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    }
  } else if (rank == 4) {
    total_elements =
        N * (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (strcmp(mode, "constant") == 0) {
      dinoml::pad_kernel_constant_4d_nhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          C,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right,
          pad_value);
    } else if (strcmp(mode, "reflect") == 0) {
      dinoml::pad_kernel_reflect_4d_nhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          C,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "replicate") == 0) {
      dinoml::pad_kernel_replicate_4d_nhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          C,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "circular") == 0) {
      dinoml::pad_kernel_circular_4d_nhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          H,
          W,
          C,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    }
  } else if (rank == 5) {
    IndexType total_elements = N * (D + pad_front + pad_back) *
        (H + pad_top + pad_bottom) * (W + pad_left + pad_right) * C;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    if (strcmp(mode, "constant") == 0) {
      dinoml::pad_kernel_constant_5d_ndhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          D,
          H,
          W,
          C,
          pad_front,
          pad_back,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right,
          pad_value);
    } else if (strcmp(mode, "reflect") == 0) {
      dinoml::pad_kernel_reflect_5d_ndhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          D,
          H,
          W,
          C,
          pad_front,
          pad_back,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "replicate") == 0) {
      dinoml::pad_kernel_replicate_5d_ndhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          D,
          H,
          W,
          C,
          pad_front,
          pad_back,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    } else if (strcmp(mode, "circular") == 0) {
      dinoml::pad_kernel_circular_5d_ndhwc<<<
          num_blocks,
          threads_per_block,
          0,
          stream>>>(
          static_cast<const ElemInputType*>(in_ptr),
          static_cast<ElemOutputType*>(out_ptr),
          N,
          D,
          H,
          W,
          C,
          pad_front,
          pad_back,
          pad_top,
          pad_bottom,
          pad_left,
          pad_right);
    }
  }
}