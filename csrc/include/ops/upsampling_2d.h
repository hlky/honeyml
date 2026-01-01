#include <dinoml/device.h>
#include <dinoml/helpers.h>
#include <cstdint>
#include <type_traits>

namespace dinoml {

template <
    typename ElemInputType,
    typename VectorType,
    typename IndexType,
    bool AlignCorners>
__global__ void bilinear_upsampling_2d_kernel(
    const ElemInputType* input_raw,
    ElemInputType* output_raw,
    const IndexType batch,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_height,
    const IndexType out_width) {
  const VectorType* input = (const VectorType*)input_raw;
  VectorType* output = (VectorType*)output_raw;

  float height_scale;
  float width_scale;
  if (AlignCorners) {
    height_scale =
        in_height == 1 ? 0.0f : (float)(in_height - 1) / (out_height - 1);
    width_scale =
        in_width == 1 ? 0.0f : (float)(in_width - 1) / (out_width - 1);
  } else {
    height_scale = in_height / static_cast<float>(out_height);
    width_scale = in_width / static_cast<float>(out_width);
  }

  const int64_t num_threads = out_height * out_width * channels * batch;

  for (int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
       out_idx < num_threads;
       out_idx += blockDim.x * gridDim.x) {
    int64_t idx = out_idx;
    const int64_t c = idx % channels;
    idx /= channels;
    const int64_t x = idx % out_width;
    idx /= out_width;
    const int64_t y = idx % out_height;
    const int64_t b = idx / out_height;

    float in_y;
    if (AlignCorners) {
      in_y = y * height_scale;
    } else {
      in_y = (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    }
    const int64_t top_y_index = in_y > 0.0 ? floorf(in_y) : 0;
    const int64_t bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    float in_x;
    if (AlignCorners) {
      in_x = x * width_scale;
    } else {
      in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    }
    const int64_t left_x_index = in_x > 0.0 ? floorf(in_x) : 0;
    const int64_t right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    const VectorType top_left = LDG(
        input +
        ((b * in_height + top_y_index) * in_width + left_x_index) * channels +
        c);

    const VectorType top_right = LDG(
        input +
        ((b * in_height + top_y_index) * in_width + right_x_index) * channels +
        c);
    const VectorType bottom_left =
        LDG(input +
            ((b * in_height + bottom_y_index) * in_width + left_x_index) *
                channels +
            c);
    const VectorType bottom_right =
        LDG(input +
            ((b * in_height + bottom_y_index) * in_width + right_x_index) *
                channels +
            c);

    const float2 tl =
        dinoml::helpers::convert<float2, VectorType>::run(top_left);
    const float2 tr =
        dinoml::helpers::convert<float2, VectorType>::run(top_right);
    const float2 bl =
        dinoml::helpers::convert<float2, VectorType>::run(bottom_left);
    const float2 br =
        dinoml::helpers::convert<float2, VectorType>::run(bottom_right);

    const float top_x = tl.x + (tr.x - tl.x) * x_lerp;
    const float top_y = tl.y + (tr.y - tl.y) * x_lerp;
    const float bottom_x = bl.x + (br.x - bl.x) * x_lerp;
    const float bottom_y = bl.y + (br.y - bl.y) * x_lerp;

    float2 out = {0.f, 0.f};
    out.x = top_x + (bottom_x - top_x) * y_lerp;
    out.y = top_y + (bottom_y - top_y) * y_lerp;

    output[out_idx] = dinoml::helpers::convert<VectorType, float2>::run(out);
  }
}

template <
    typename ElemInputType,
    typename VectorType,
    typename IndexType,
    bool AlignCorners>
__global__ void bilinear_upsampling_2d_add_kernel(
    const ElemInputType* input_raw,
    const ElemInputType* input_res_raw,
    ElemInputType* output_raw,
    const IndexType batch,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_height,
    const IndexType out_width) {
  const VectorType* input = (const VectorType*)input_raw;
  VectorType* output = (VectorType*)output_raw;
  const VectorType* input_res = (const VectorType*)input_res_raw;

  float height_scale;
  float width_scale;
  if (AlignCorners) {
    height_scale =
        in_height == 1 ? 0.0f : (float)(in_height - 1) / (out_height - 1);
    width_scale =
        in_width == 1 ? 0.0f : (float)(in_width - 1) / (out_width - 1);
  } else {
    height_scale = in_height / static_cast<float>(out_height);
    width_scale = in_width / static_cast<float>(out_width);
  }

  const int64_t num_threads = out_height * out_width * channels * batch;

  for (int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
       out_idx < num_threads;
       out_idx += blockDim.x * gridDim.x) {
    int64_t idx = out_idx;
    const int64_t c = idx % channels;
    idx /= channels;
    const int64_t x = idx % out_width;
    idx /= out_width;
    const int64_t y = idx % out_height;
    const int64_t b = idx / out_height;

    float in_y;
    if (AlignCorners) {
      in_y = y * height_scale;
    } else {
      in_y = (static_cast<float>(y) + 0.5f) * height_scale - 0.5f;
    }
    const int64_t top_y_index = in_y > 0.0 ? floorf(in_y) : 0;
    const int64_t bottom_y_index =
        (in_y < in_height - 1) ? ceilf(in_y) : in_height - 1;
    const float y_lerp = in_y - floorf(in_y);

    float in_x;
    if (AlignCorners) {
      in_x = x * width_scale;
    } else {
      in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    }
    const int64_t left_x_index = in_x > 0.0 ? floorf(in_x) : 0;
    const int64_t right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    const VectorType top_left = LDG(
        input +
        ((b * in_height + top_y_index) * in_width + left_x_index) * channels +
        c);

    const VectorType top_right = LDG(
        input +
        ((b * in_height + top_y_index) * in_width + right_x_index) * channels +
        c);
    const VectorType bottom_left =
        LDG(input +
            ((b * in_height + bottom_y_index) * in_width + left_x_index) *
                channels +
            c);
    const VectorType bottom_right =
        LDG(input +
            ((b * in_height + bottom_y_index) * in_width + right_x_index) *
                channels +
            c);

    const float2 tl =
        dinoml::helpers::convert<float2, VectorType>::run(top_left);
    const float2 tr =
        dinoml::helpers::convert<float2, VectorType>::run(top_right);
    const float2 bl =
        dinoml::helpers::convert<float2, VectorType>::run(bottom_left);
    const float2 br =
        dinoml::helpers::convert<float2, VectorType>::run(bottom_right);

    const float top_x = tl.x + (tr.x - tl.x) * x_lerp;
    const float top_y = tl.y + (tr.y - tl.y) * x_lerp;
    const float bottom_x = bl.x + (br.x - bl.x) * x_lerp;
    const float bottom_y = bl.y + (br.y - bl.y) * x_lerp;

    float2 out = {0.f, 0.f};
    out.x = top_x + (bottom_x - top_x) * y_lerp;
    out.y = top_y + (bottom_y - top_y) * y_lerp;

    output[out_idx] = dinoml::helpers::add2(LDG(input_res + out_idx), out);
  }
}

template <
    typename ElemInputType,
    typename VectorType,
    typename IndexType,
    int VectorSize,
    bool Exact>
__global__ void nearest_upsampling_2d_kernel(
    const VectorType* input,
    VectorType* output,
    const IndexType batch,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_height,
    const IndexType out_width) {
  const float height_scale = in_height / static_cast<float>(out_height);
  const float width_scale = in_width / static_cast<float>(out_width);
  const int64_t num_threads = out_height * out_width * channels * batch;

  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < num_threads;
       index += blockDim.x * gridDim.x) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const VectorType* bottom_data_n =
        input + n * channels * in_height * in_width;
    int in_y;
    int in_x;
    if (Exact) {
      in_y = max(
          min(static_cast<int>(
                  floorf((static_cast<float>(out_y) + 0.5f) * height_scale)),
              static_cast<int>(in_height) - 1),
          0);
      in_x =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_x) + 0.5f) * width_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    } else {
      in_y =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_y)) * height_scale)),
                  static_cast<int>(in_height) - 1),
              0);
      in_x =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_x)) * width_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    }

    const int idx = (in_y * in_width + in_x) * channels + c;

    output[index] = LDG(bottom_data_n + idx);
  }
}

template <
    typename ElemType,
    typename VectorType,
    typename IndexType,
    int VectorSize,
    bool Exact>
__global__ void nearest_upsampling_2d_add_kernel(
    const VectorType* input,
    const VectorType* input_res,
    VectorType* output,
    const IndexType batch,
    const IndexType in_height,
    const IndexType in_width,
    const IndexType channels,
    const IndexType out_height,
    const IndexType out_width) {
  const float height_scale = in_height / static_cast<float>(out_height);
  const float width_scale = in_width / static_cast<float>(out_width);
  const int64_t num_threads = out_height * out_width * channels * batch;

  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < num_threads;
       index += blockDim.x * gridDim.x) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const VectorType* bottom_data_n =
        input + n * channels * in_height * in_width;
    int in_y;
    int in_x;
    if (Exact) {
      in_y = max(
          min(static_cast<int>(
                  floorf((static_cast<float>(out_y) + 0.5f) * height_scale)),
              static_cast<int>(in_height) - 1),
          0);
      in_x =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_x) + 0.5f) * width_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    } else {
      in_y =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_y)) * height_scale)),
                  static_cast<int>(in_height) - 1),
              0);
      in_x =
          max(min(static_cast<int>(
                      floorf((static_cast<float>(out_x)) * width_scale)),
                  static_cast<int>(in_width) - 1),
              0);
    }

    const int idx = (in_y * in_width + in_x) * channels + c;

    VectorType input_val = LDG(bottom_data_n + idx);
    VectorType input_res_val = LDG(input_res + index);
    output[index] = dinoml::helpers::add_op2<VectorType, ElemType>::run(
        input_val, input_res_val);
  }
}

enum class Upsampling2dMode {
  BILINEAR = 0,
  NEAREST = 1,
  NEAREST_EXACT = 2,
};

} // namespace dinoml

template <
    typename ElemType,
    typename VectorType,
    typename IndexType,
    int Alignment,
    dinoml::Upsampling2dMode Mode,
    bool AlignCorners,
    bool Exact,
    bool HasResidual>
void upsampling_2d_launcher(
    const ElemType* input,
    const ElemType* res,
    ElemType* output,
    const IndexType N,
    const IndexType H,
    const IndexType W,
    const IndexType C,
    const IndexType HO,
    const IndexType WO,
    dinoml::DeviceStream stream) {
  const int64_t output_size = N * (C)*HO * WO;
  dim3 grid(
      std::min(
          dinoml::helpers::ceil_div(
              static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
          static_cast<int64_t>(4096)));
  dim3 block(512);

  if constexpr (Mode == dinoml::Upsampling2dMode::BILINEAR) {
    if constexpr (
        std::is_same_v<VectorType, float2> ||
        std::is_same_v<VectorType, half2> ||
        std::is_same_v<VectorType, bfloat162>) {
      if constexpr (HasResidual) {
        dinoml::bilinear_upsampling_2d_add_kernel<
            ElemType,
            VectorType,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, res, output, N, H, W, C / Alignment, HO, WO);
      } else {
        dinoml::bilinear_upsampling_2d_kernel<
            ElemType,
            VectorType,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, output, N, H, W, C / Alignment, HO, WO);
      }
    }
  } else if constexpr (
      Mode == dinoml::Upsampling2dMode::NEAREST ||
      Mode == dinoml::Upsampling2dMode::NEAREST_EXACT) {
    if constexpr (HasResidual) {
      dinoml::nearest_upsampling_2d_add_kernel<
          ElemType,
          VectorType,
          IndexType,
          Alignment,
          Exact><<<grid, block, 0, stream>>>(
          (const VectorType*)input,
          (const VectorType*)res,
          (VectorType*)output,
          N,
          H,
          W,
          C / Alignment,
          HO,
          WO);
    } else {
      dinoml::nearest_upsampling_2d_kernel<
          ElemType,
          VectorType,
          IndexType,
          Alignment,
          Exact><<<grid, block, 0, stream>>>(
          (const VectorType*)input,
          (VectorType*)output,
          N,
          H,
          W,
          C / Alignment,
          HO,
          WO);
    }
  } else {
    throw std::runtime_error("Unsupported upsampling mode.");
  }
}