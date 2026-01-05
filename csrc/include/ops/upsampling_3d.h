#pragma once

#include <dinoml/device.h>
#include <dinoml/helpers.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace dinoml {

template <
    typename ElemInputType,
    typename VectorType,
    typename IndexType,
    bool AlignCorners>
__global__ void trilinear_upsampling_3d_kernel(
    const ElemInputType* input_raw,
    ElemInputType* output_raw,
    const IndexType batch,
    const IndexType in_frames,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_frames,
    const IndexType out_height,
    const IndexType out_width) {
  const VectorType* input = reinterpret_cast<const VectorType*>(input_raw);
  VectorType* output = reinterpret_cast<VectorType*>(output_raw);

  float f_scale, h_scale, w_scale;
  if (AlignCorners) {
    f_scale = (in_frames == 1)
        ? 0.0f
        : static_cast<float>(in_frames - 1) / (out_frames - 1);
    h_scale = (in_height == 1)
        ? 0.0f
        : static_cast<float>(in_height - 1) / (out_height - 1);
    w_scale = (in_width == 1)
        ? 0.0f
        : static_cast<float>(in_width - 1) / (out_width - 1);
  } else {
    f_scale = static_cast<float>(in_frames) / static_cast<float>(out_frames);
    h_scale = static_cast<float>(in_height) / static_cast<float>(out_height);
    w_scale = static_cast<float>(in_width) / static_cast<float>(out_width);
  }

  const int64_t num_threads = static_cast<int64_t>(out_frames) *
      static_cast<int64_t>(out_height) * static_cast<int64_t>(out_width) *
      static_cast<int64_t>(channels) * static_cast<int64_t>(batch);

  for (int64_t out_idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       out_idx < num_threads;
       out_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t idx = out_idx;

    const int64_t c = idx % channels;
    idx /= channels;
    const int64_t x = idx % out_width;
    idx /= out_width;
    const int64_t y = idx % out_height;
    idx /= out_height;
    const int64_t f = idx % out_frames;
    const int64_t b = idx / out_frames;

    float in_f, in_y, in_x;

    if (AlignCorners) {
      in_f = static_cast<float>(f) * f_scale;
      in_y = static_cast<float>(y) * h_scale;
      in_x = static_cast<float>(x) * w_scale;
    } else {
      in_f = (static_cast<float>(f) + 0.5f) * f_scale - 0.5f;
      in_y = (static_cast<float>(y) + 0.5f) * h_scale - 0.5f;
      in_x = (static_cast<float>(x) + 0.5f) * w_scale - 0.5f;
    }

    const int64_t f0 = (in_f > 0.0f) ? static_cast<int64_t>(floorf(in_f)) : 0;
    const int64_t f1 = (in_f < in_frames - 1)
        ? static_cast<int64_t>(ceilf(in_f))
        : in_frames - 1;
    const float f_lerp = in_f - floorf(in_f);

    const int64_t y0 = (in_y > 0.0f) ? static_cast<int64_t>(floorf(in_y)) : 0;
    const int64_t y1 = (in_y < in_height - 1)
        ? static_cast<int64_t>(ceilf(in_y))
        : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    const int64_t x0 = (in_x > 0.0f) ? static_cast<int64_t>(floorf(in_x)) : 0;
    const int64_t x1 = (in_x < in_width - 1) ? static_cast<int64_t>(ceilf(in_x))
                                             : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    // Input layout: ((b*F + f)*H + y)*W + x)*C + c
    const int64_t base0 =
        (((b * in_frames + f0) * in_height + y0) * in_width + x0) * channels +
        c;
    const int64_t base1 =
        (((b * in_frames + f0) * in_height + y0) * in_width + x1) * channels +
        c;
    const int64_t base2 =
        (((b * in_frames + f0) * in_height + y1) * in_width + x0) * channels +
        c;
    const int64_t base3 =
        (((b * in_frames + f0) * in_height + y1) * in_width + x1) * channels +
        c;

    const int64_t base4 =
        (((b * in_frames + f1) * in_height + y0) * in_width + x0) * channels +
        c;
    const int64_t base5 =
        (((b * in_frames + f1) * in_height + y0) * in_width + x1) * channels +
        c;
    const int64_t base6 =
        (((b * in_frames + f1) * in_height + y1) * in_width + x0) * channels +
        c;
    const int64_t base7 =
        (((b * in_frames + f1) * in_height + y1) * in_width + x1) * channels +
        c;

    const VectorType v000 = LDG(input + base0);
    const VectorType v001 = LDG(input + base1);
    const VectorType v010 = LDG(input + base2);
    const VectorType v011 = LDG(input + base3);

    const VectorType v100 = LDG(input + base4);
    const VectorType v101 = LDG(input + base5);
    const VectorType v110 = LDG(input + base6);
    const VectorType v111 = LDG(input + base7);

    const float2 f000 = dinoml::helpers::convert<float2, VectorType>::run(v000);
    const float2 f001 = dinoml::helpers::convert<float2, VectorType>::run(v001);
    const float2 f010 = dinoml::helpers::convert<float2, VectorType>::run(v010);
    const float2 f011 = dinoml::helpers::convert<float2, VectorType>::run(v011);

    const float2 f100 = dinoml::helpers::convert<float2, VectorType>::run(v100);
    const float2 f101 = dinoml::helpers::convert<float2, VectorType>::run(v101);
    const float2 f110 = dinoml::helpers::convert<float2, VectorType>::run(v110);
    const float2 f111 = dinoml::helpers::convert<float2, VectorType>::run(v111);

    // Interp in x
    auto lerp2 = [](const float2& a, const float2& b, float t) -> float2 {
      return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
    };

    const float2 f00 = lerp2(f000, f001, x_lerp);
    const float2 f01 = lerp2(f010, f011, x_lerp);
    const float2 f10 = lerp2(f100, f101, x_lerp);
    const float2 f11 = lerp2(f110, f111, x_lerp);

    const float2 f0y = lerp2(f00, f01, y_lerp);
    const float2 f1y = lerp2(f10, f11, y_lerp);

    const float2 out_f = lerp2(f0y, f1y, f_lerp);

    output[out_idx] = dinoml::helpers::convert<VectorType, float2>::run(out_f);
  }
}

template <
    typename ElemInputType,
    typename VectorType,
    typename IndexType,
    bool AlignCorners>
__global__ void trilinear_upsampling_3d_add_kernel(
    const ElemInputType* input_raw,
    const ElemInputType* input_res_raw,
    ElemInputType* output_raw,
    const IndexType batch,
    const IndexType in_frames,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_frames,
    const IndexType out_height,
    const IndexType out_width) {
  const VectorType* input = reinterpret_cast<const VectorType*>(input_raw);
  const VectorType* input_res =
      reinterpret_cast<const VectorType*>(input_res_raw);
  VectorType* output = reinterpret_cast<VectorType*>(output_raw);

  float f_scale, h_scale, w_scale;
  if (AlignCorners) {
    f_scale = (in_frames == 1)
        ? 0.0f
        : static_cast<float>(in_frames - 1) / (out_frames - 1);
    h_scale = (in_height == 1)
        ? 0.0f
        : static_cast<float>(in_height - 1) / (out_height - 1);
    w_scale = (in_width == 1)
        ? 0.0f
        : static_cast<float>(in_width - 1) / (out_width - 1);
  } else {
    f_scale = static_cast<float>(in_frames) / static_cast<float>(out_frames);
    h_scale = static_cast<float>(in_height) / static_cast<float>(out_height);
    w_scale = static_cast<float>(in_width) / static_cast<float>(out_width);
  }

  const int64_t num_threads = static_cast<int64_t>(out_frames) *
      static_cast<int64_t>(out_height) * static_cast<int64_t>(out_width) *
      static_cast<int64_t>(channels) * static_cast<int64_t>(batch);

  for (int64_t out_idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       out_idx < num_threads;
       out_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t idx = out_idx;

    const int64_t c = idx % channels;
    idx /= channels;
    const int64_t x = idx % out_width;
    idx /= out_width;
    const int64_t y = idx % out_height;
    idx /= out_height;
    const int64_t f = idx % out_frames;
    const int64_t b = idx / out_frames;

    float in_f, in_y, in_x;

    if (AlignCorners) {
      in_f = static_cast<float>(f) * f_scale;
      in_y = static_cast<float>(y) * h_scale;
      in_x = static_cast<float>(x) * w_scale;
    } else {
      in_f = (static_cast<float>(f) + 0.5f) * f_scale - 0.5f;
      in_y = (static_cast<float>(y) + 0.5f) * h_scale - 0.5f;
      in_x = (static_cast<float>(x) + 0.5f) * w_scale - 0.5f;
    }

    const int64_t f0 = (in_f > 0.0f) ? static_cast<int64_t>(floorf(in_f)) : 0;
    const int64_t f1 = (in_f < in_frames - 1)
        ? static_cast<int64_t>(ceilf(in_f))
        : in_frames - 1;
    const float f_lerp = in_f - floorf(in_f);

    const int64_t y0 = (in_y > 0.0f) ? static_cast<int64_t>(floorf(in_y)) : 0;
    const int64_t y1 = (in_y < in_height - 1)
        ? static_cast<int64_t>(ceilf(in_y))
        : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    const int64_t x0 = (in_x > 0.0f) ? static_cast<int64_t>(floorf(in_x)) : 0;
    const int64_t x1 = (in_x < in_width - 1) ? static_cast<int64_t>(ceilf(in_x))
                                             : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    const int64_t base0 =
        (((b * in_frames + f0) * in_height + y0) * in_width + x0) * channels +
        c;
    const int64_t base1 =
        (((b * in_frames + f0) * in_height + y0) * in_width + x1) * channels +
        c;
    const int64_t base2 =
        (((b * in_frames + f0) * in_height + y1) * in_width + x0) * channels +
        c;
    const int64_t base3 =
        (((b * in_frames + f0) * in_height + y1) * in_width + x1) * channels +
        c;

    const int64_t base4 =
        (((b * in_frames + f1) * in_height + y0) * in_width + x0) * channels +
        c;
    const int64_t base5 =
        (((b * in_frames + f1) * in_height + y0) * in_width + x1) * channels +
        c;
    const int64_t base6 =
        (((b * in_frames + f1) * in_height + y1) * in_width + x0) * channels +
        c;
    const int64_t base7 =
        (((b * in_frames + f1) * in_height + y1) * in_width + x1) * channels +
        c;

    const VectorType v000 = LDG(input + base0);
    const VectorType v001 = LDG(input + base1);
    const VectorType v010 = LDG(input + base2);
    const VectorType v011 = LDG(input + base3);

    const VectorType v100 = LDG(input + base4);
    const VectorType v101 = LDG(input + base5);
    const VectorType v110 = LDG(input + base6);
    const VectorType v111 = LDG(input + base7);

    const float2 f000 = dinoml::helpers::convert<float2, VectorType>::run(v000);
    const float2 f001 = dinoml::helpers::convert<float2, VectorType>::run(v001);
    const float2 f010 = dinoml::helpers::convert<float2, VectorType>::run(v010);
    const float2 f011 = dinoml::helpers::convert<float2, VectorType>::run(v011);

    const float2 f100 = dinoml::helpers::convert<float2, VectorType>::run(v100);
    const float2 f101 = dinoml::helpers::convert<float2, VectorType>::run(v101);
    const float2 f110 = dinoml::helpers::convert<float2, VectorType>::run(v110);
    const float2 f111 = dinoml::helpers::convert<float2, VectorType>::run(v111);

    auto lerp2 = [](const float2& a, const float2& b, float t) -> float2 {
      return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
    };

    const float2 f00 = lerp2(f000, f001, x_lerp);
    const float2 f01 = lerp2(f010, f011, x_lerp);
    const float2 f10 = lerp2(f100, f101, x_lerp);
    const float2 f11 = lerp2(f110, f111, x_lerp);

    const float2 f0y = lerp2(f00, f01, y_lerp);
    const float2 f1y = lerp2(f10, f11, y_lerp);

    const float2 out_f = lerp2(f0y, f1y, f_lerp);

    output[out_idx] = dinoml::helpers::add2(LDG(input_res + out_idx), out_f);
  }
}

template <
    typename ElemType,
    typename VectorType,
    typename IndexType,
    int VectorSize,
    bool Exact>
__global__ void nearest_upsampling_3d_kernel(
    const VectorType* input,
    VectorType* output,
    const IndexType batch,
    const IndexType in_frames,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_frames,
    const IndexType out_height,
    const IndexType out_width) {
  const float f_scale =
      static_cast<float>(in_frames) / static_cast<float>(out_frames);
  const float h_scale =
      static_cast<float>(in_height) / static_cast<float>(out_height);
  const float w_scale =
      static_cast<float>(in_width) / static_cast<float>(out_width);

  const int64_t num_threads = static_cast<int64_t>(out_frames) *
      static_cast<int64_t>(out_height) * static_cast<int64_t>(out_width) *
      static_cast<int64_t>(channels) * static_cast<int64_t>(batch);

  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_threads;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t n = index;

    const int c = static_cast<int>(n % channels);
    n /= channels;
    const int out_x = static_cast<int>(n % out_width);
    n /= out_width;
    const int out_y = static_cast<int>(n % out_height);
    n /= out_height;
    const int out_f = static_cast<int>(n % out_frames);
    n /= out_frames;

    const VectorType* base_n =
        input + n * channels * in_frames * in_height * in_width;

    int in_f, in_y, in_x;
    if (Exact) {
      in_f =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_f) + 0.5f) * f_scale)),
                  static_cast<int>(in_frames) - 1),
              0);
      in_y =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_y) + 0.5f) * h_scale)),
                  static_cast<int>(in_height) - 1),
              0);
      in_x =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_x) + 0.5f) * w_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    } else {
      in_f =
          max(min(static_cast<int>(floorf(static_cast<float>(out_f) * f_scale)),
                  static_cast<int>(in_frames) - 1),
              0);
      in_y =
          max(min(static_cast<int>(floorf(static_cast<float>(out_y) * h_scale)),
                  static_cast<int>(in_height) - 1),
              0);
      in_x =
          max(min(static_cast<int>(floorf(static_cast<float>(out_x) * w_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    }

    const int idx = (((in_f * static_cast<int>(in_height) + in_y) *
                          static_cast<int>(in_width) +
                      in_x) *
                     static_cast<int>(channels)) +
        c;

    output[index] = LDG(base_n + idx);
  }
}

template <
    typename ElemType,
    typename VectorType,
    typename IndexType,
    int VectorSize,
    bool Exact>
__global__ void nearest_upsampling_3d_kernel_strided(
    const VectorType* input,
    VectorType* output,
    const IndexType batch,
    const IndexType in_frames,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_frames,
    const IndexType out_height,
    const IndexType out_width,
    const int64_t in_batch_stride,
    const int64_t out_batch_stride
) {
  const float f_scale = (float)in_frames / (float)out_frames;
  const float h_scale = (float)in_height / (float)out_height;
  const float w_scale = (float)in_width / (float)out_width;

  const int64_t num_threads =
      (int64_t)batch * out_frames * out_height * out_width * channels;

  for (int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < num_threads;
       index += (int64_t)blockDim.x * gridDim.x) {
    int64_t t = index;

    const int c = (int)(t % channels);
    t /= channels;
    const int out_x = (int)(t % out_width);
    t /= out_width;
    const int out_y = (int)(t % out_height);
    t /= out_height;
    const int out_f = (int)(t % out_frames);
    t /= out_frames;
    const int n = (int)t;

    const VectorType* base_n = input + (int64_t)n * in_batch_stride;
    VectorType* out_base_n = output + (int64_t)n * out_batch_stride;

    int in_f, in_y, in_x;
    if (Exact) {
      in_f = max(
          min((int)floorf(((float)out_f + 0.5f) * f_scale), (int)in_frames - 1),
          0);
      in_y = max(
          min((int)floorf(((float)out_y + 0.5f) * h_scale), (int)in_height - 1),
          0);
      in_x = max(
          min((int)floorf(((float)out_x + 0.5f) * w_scale), (int)in_width - 1),
          0);
    } else {
      in_f =
          max(min((int)floorf((float)out_f * f_scale), (int)in_frames - 1), 0);
      in_y =
          max(min((int)floorf((float)out_y * h_scale), (int)in_height - 1), 0);
      in_x =
          max(min((int)floorf((float)out_x * w_scale), (int)in_width - 1), 0);
    }

    const int64_t in_idx =
        ((((int64_t)in_f * in_height + in_y) * in_width + in_x) * channels) + c;

    out_base_n
        [index - (int64_t)n * out_frames * out_height * out_width * channels] =
            LDG(base_n + in_idx);
  }
}

template <
    typename ElemType,
    typename VectorType,
    typename IndexType,
    int VectorSize,
    bool Exact>
__global__ void nearest_upsampling_3d_add_kernel(
    const VectorType* input,
    const VectorType* input_res,
    VectorType* output,
    const IndexType batch,
    const IndexType in_frames,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_frames,
    const IndexType out_height,
    const IndexType out_width) {
  const float f_scale =
      static_cast<float>(in_frames) / static_cast<float>(out_frames);
  const float h_scale =
      static_cast<float>(in_height) / static_cast<float>(out_height);
  const float w_scale =
      static_cast<float>(in_width) / static_cast<float>(out_width);

  const int64_t num_threads = static_cast<int64_t>(out_frames) *
      static_cast<int64_t>(out_height) * static_cast<int64_t>(out_width) *
      static_cast<int64_t>(channels) * static_cast<int64_t>(batch);

  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_threads;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t n = index;

    const int c = static_cast<int>(n % channels);
    n /= channels;
    const int out_x = static_cast<int>(n % out_width);
    n /= out_width;
    const int out_y = static_cast<int>(n % out_height);
    n /= out_height;
    const int out_f = static_cast<int>(n % out_frames);
    n /= out_frames;

    const VectorType* base_n =
        input + n * channels * in_frames * in_height * in_width;

    int in_f, in_y, in_x;
    if (Exact) {
      in_f =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_f) + 0.5f) * f_scale)),
                  static_cast<int>(in_frames) - 1),
              0);
      in_y =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_y) + 0.5f) * h_scale)),
                  static_cast<int>(in_height) - 1),
              0);
      in_x =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_x) + 0.5f) * w_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    } else {
      in_f =
          max(min(static_cast<int>(floorf(static_cast<float>(out_f) * f_scale)),
                  static_cast<int>(in_frames) - 1),
              0);
      in_y =
          max(min(static_cast<int>(floorf(static_cast<float>(out_y) * h_scale)),
                  static_cast<int>(in_height) - 1),
              0);
      in_x =
          max(min(static_cast<int>(floorf(static_cast<float>(out_x) * w_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    }

    const int idx = (((in_f * static_cast<int>(in_height) + in_y) *
                          static_cast<int>(in_width) +
                      in_x) *
                     static_cast<int>(channels)) +
        c;

    VectorType input_val = LDG(base_n + idx);
    VectorType res_val = LDG(input_res + index);

    output[index] =
        dinoml::helpers::add_op2<VectorType, ElemType>::run(input_val, res_val);
  }
}

enum class Upsampling3dMode {
  TRILINEAR = 0,
  NEAREST = 1,
  NEAREST_EXACT = 2,
};

} // namespace dinoml

template <
    typename ElemType,
    typename VectorType,
    typename IndexType,
    int Alignment,
    dinoml::Upsampling3dMode Mode,
    bool AlignCorners,
    bool Exact,
    bool HasResidual>
void upsampling_3d_launcher(
    const ElemType* input,
    const ElemType* res,
    ElemType* output,
    const IndexType N,
    const IndexType F,
    const IndexType H,
    const IndexType W,
    const IndexType C,
    const IndexType FO,
    const IndexType HO,
    const IndexType WO,
    dinoml::DeviceStream stream) {
  const int64_t output_size = static_cast<int64_t>(N) *
      static_cast<int64_t>(C) * static_cast<int64_t>(FO) *
      static_cast<int64_t>(HO) * static_cast<int64_t>(WO);

  dim3 grid(
      std::min(
          dinoml::helpers::ceil_div(
              static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
          static_cast<int64_t>(4096)));
  dim3 block(512);

  if constexpr (Mode == dinoml::Upsampling3dMode::TRILINEAR) {
    if constexpr (
        std::is_same_v<VectorType, float2> ||
        std::is_same_v<VectorType, half2> ||
        std::is_same_v<VectorType, bfloat162>) {
      if constexpr (HasResidual) {
        dinoml::trilinear_upsampling_3d_add_kernel<
            ElemType,
            VectorType,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, res, output, N, F, H, W, C / Alignment, FO, HO, WO);
      } else {
        dinoml::trilinear_upsampling_3d_kernel<
            ElemType,
            VectorType,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, output, N, F, H, W, C / Alignment, FO, HO, WO);
      }
    }
  } else if constexpr (
      Mode == dinoml::Upsampling3dMode::NEAREST ||
      Mode == dinoml::Upsampling3dMode::NEAREST_EXACT) {
    if constexpr (HasResidual) {
      dinoml::nearest_upsampling_3d_add_kernel<
          ElemType,
          VectorType,
          IndexType,
          Alignment,
          Exact><<<grid, block, 0, stream>>>(
          reinterpret_cast<const VectorType*>(input),
          reinterpret_cast<const VectorType*>(res),
          reinterpret_cast<VectorType*>(output),
          N,
          F,
          H,
          W,
          C / Alignment,
          FO,
          HO,
          WO);
    } else {
      dinoml::nearest_upsampling_3d_kernel<
          ElemType,
          VectorType,
          IndexType,
          Alignment,
          Exact><<<grid, block, 0, stream>>>(
          reinterpret_cast<const VectorType*>(input),
          reinterpret_cast<VectorType*>(output),
          N,
          F,
          H,
          W,
          C / Alignment,
          FO,
          HO,
          WO);
    }
  } else {
    throw std::runtime_error("Unsupported upsampling mode.");
  }
}
