#include <dinoml/device.h>
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

    float top_x;
    float top_y;
    float bottom_x;
    float bottom_y;
    if constexpr (std::is_same_v<ElemInputType, float>) {
      top_x = HALF2DATA(top_left).x +
          (HALF2DATA(top_right).x - HALF2DATA(top_left).x) * x_lerp;
      top_y = HALF2DATA(top_left).y +
          (HALF2DATA(top_right).y - HALF2DATA(top_left).y) * x_lerp;
      bottom_x = HALF2DATA(bottom_left).x +
          (HALF2DATA(bottom_right).x - HALF2DATA(bottom_left).x) * x_lerp;
      bottom_y = HALF2DATA(bottom_left).y +
          (HALF2DATA(bottom_right).y - HALF2DATA(bottom_left).y) * x_lerp;
    } else if constexpr (std::is_same_v<ElemInputType, half>) {
      top_x = __half2float(HALF2DATA(top_left).x) +
          (__half2float(HALF2DATA(top_right).x) -
           __half2float(HALF2DATA(top_left).x)) *
              x_lerp;
      top_y = __half2float(HALF2DATA(top_left).y) +
          (__half2float(HALF2DATA(top_right).y) -
           __half2float(HALF2DATA(top_left).y)) *
              x_lerp;
      bottom_x = __half2float(HALF2DATA(bottom_left).x) +
          (__half2float(HALF2DATA(bottom_right).x) -
           __half2float(HALF2DATA(bottom_left).x)) *
              x_lerp;
      bottom_y = __half2float(HALF2DATA(bottom_left).y) +
          (__half2float(HALF2DATA(bottom_right).y) -
           __half2float(HALF2DATA(bottom_left).y)) *
              x_lerp;
    } else if constexpr (std::is_same_v<ElemInputType, bfloat16>) {
      top_x = __bfloat162float(HALF2DATA(top_left).x) +
          (__bfloat162float(HALF2DATA(top_right).x) -
           __bfloat162float(HALF2DATA(top_left).x)) *
              x_lerp;
      top_y = __bfloat162float(HALF2DATA(top_left).y) +
          (__bfloat162float(HALF2DATA(top_right).y) -
           __bfloat162float(HALF2DATA(top_left).y)) *
              x_lerp;
      bottom_x = __bfloat162float(HALF2DATA(bottom_left).x) +
          (__bfloat162float(HALF2DATA(bottom_right).x) -
           __bfloat162float(HALF2DATA(bottom_left).x)) *
              x_lerp;
      bottom_y = __bfloat162float(HALF2DATA(bottom_left).y) +
          (__bfloat162float(HALF2DATA(bottom_right).y) -
           __bfloat162float(HALF2DATA(bottom_left).y)) *
              x_lerp;
    }

    float2 out = {0.f, 0.f};
    out.x = top_x + (bottom_x - top_x) * y_lerp;
    out.y = top_y + (bottom_y - top_y) * y_lerp;

    if constexpr (std::is_same_v<ElemInputType, float>) {
      output[out_idx] = out;
    } else if constexpr (std::is_same_v<ElemInputType, half>) {
      output[out_idx] = __float22half2_rn(out);
    } else if constexpr (std::is_same_v<ElemInputType, bfloat16>) {
      output[out_idx] = __float22bfloat162_rn(out);
    }
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

    float top_x;
    float top_y;
    float bottom_x;
    float bottom_y;
    if constexpr (std::is_same_v<ElemInputType, float>) {
      top_x = HALF2DATA(top_left).x +
          (HALF2DATA(top_right).x - HALF2DATA(top_left).x) * x_lerp;
      top_y = HALF2DATA(top_left).y +
          (HALF2DATA(top_right).y - HALF2DATA(top_left).y) * x_lerp;
      bottom_x = HALF2DATA(bottom_left).x +
          (HALF2DATA(bottom_right).x - HALF2DATA(bottom_left).x) * x_lerp;
      bottom_y = HALF2DATA(bottom_left).y +
          (HALF2DATA(bottom_right).y - HALF2DATA(bottom_left).y) * x_lerp;
    } else if constexpr (std::is_same_v<ElemInputType, half>) {
      top_x = __half2float(HALF2DATA(top_left).x) +
          (__half2float(HALF2DATA(top_right).x) -
           __half2float(HALF2DATA(top_left).x)) *
              x_lerp;
      top_y = __half2float(HALF2DATA(top_left).y) +
          (__half2float(HALF2DATA(top_right).y) -
           __half2float(HALF2DATA(top_left).y)) *
              x_lerp;
      bottom_x = __half2float(HALF2DATA(bottom_left).x) +
          (__half2float(HALF2DATA(bottom_right).x) -
           __half2float(HALF2DATA(bottom_left).x)) *
              x_lerp;
      bottom_y = __half2float(HALF2DATA(bottom_left).y) +
          (__half2float(HALF2DATA(bottom_right).y) -
           __half2float(HALF2DATA(bottom_left).y)) *
              x_lerp;
    } else if constexpr (std::is_same_v<ElemInputType, bfloat16>) {
      top_x = __bfloat162float(HALF2DATA(top_left).x) +
          (__bfloat162float(HALF2DATA(top_right).x) -
           __bfloat162float(HALF2DATA(top_left).x)) *
              x_lerp;
      top_y = __bfloat162float(HALF2DATA(top_left).y) +
          (__bfloat162float(HALF2DATA(top_right).y) -
           __bfloat162float(HALF2DATA(top_left).y)) *
              x_lerp;
      bottom_x = __bfloat162float(HALF2DATA(bottom_left).x) +
          (__bfloat162float(HALF2DATA(bottom_right).x) -
           __bfloat162float(HALF2DATA(bottom_left).x)) *
              x_lerp;
      bottom_y = __bfloat162float(HALF2DATA(bottom_left).y) +
          (__bfloat162float(HALF2DATA(bottom_right).y) -
           __bfloat162float(HALF2DATA(bottom_left).y)) *
              x_lerp;
    }

    float2 out = {0.f, 0.f};
    out.x = top_x + (bottom_x - top_x) * y_lerp;
    out.y = top_y + (bottom_y - top_y) * y_lerp;

    if constexpr (std::is_same_v<ElemInputType, float>) {
      const auto tmp = LDG(input_res + out_idx);
      out.x += tmp.x;
      out.y += tmp.y;
      output[out_idx] = out;
    } else if constexpr (std::is_same_v<ElemInputType, half>) {
      output[out_idx] =
          __hadd2(__float22half2_rn(out), LDG(input_res + out_idx));
    } else if constexpr (std::is_same_v<ElemInputType, bfloat16>) {
      output[out_idx] =
          __hadd2(__float22bfloat162_rn(out), LDG(input_res + out_idx));
    }
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

    VectorType input_val = __ldg(bottom_data_n + idx);
    VectorType input_res_val = __ldg(input_res + index);
    if constexpr (std::is_same_v<ElemType, VectorType>) {
      output[index] = input_val + input_res_val;
    } else {
      if (VectorSize == 8) {
        VectorType output_val;
        ElemType* pack_y = reinterpret_cast<ElemType*>(&output_val);
        ElemType* pack_x = reinterpret_cast<ElemType*>(&input_val);
        ElemType* pack_res = reinterpret_cast<ElemType*>(&input_res_val);
        for (int k = 0; k < VectorSize; k++)
          pack_y[k] = pack_x[k] + pack_res[k];
        output[index] = output_val;
      } else {
        VectorType output_val;
        HALF2DATA(output_val).x =
            HALF2DATA(input_val).x + HALF2DATA(input_res_val).x;
        HALF2DATA(output_val).y =
            HALF2DATA(input_val).y + HALF2DATA(input_res_val).y;
        output[index] = output_val;
      }
    }
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
    typename IndexType,
    int Alignment,
    dinoml::Upsampling2dMode Mode,
    bool AlignCorners,
    bool Exact>
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
          dinoml::ceil_div(
              static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
          static_cast<int64_t>(4096)));
  dim3 block(512);

  if (Mode == dinoml::Upsampling2dMode::BILINEAR) {
    if constexpr (std::is_same_v<ElemType, float>) {
      if (res == nullptr) {
        dinoml::bilinear_upsampling_2d_kernel<
            ElemType,
            float2,
            IndexType,
            AlignCorners>
            <<<grid, block, 0, stream>>>(input, output, N, H, W, C / 2, HO, WO);
      } else {
        dinoml::bilinear_upsampling_2d_add_kernel<
            ElemType,
            float2,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, res, output, N, H, W, C / 2, HO, WO);
      }
    } else if constexpr (std::is_same_v<ElemType, half>) {
      if (res == nullptr) {
        dinoml::bilinear_upsampling_2d_kernel<
            ElemType,
            half2,
            IndexType,
            AlignCorners>
            <<<grid, block, 0, stream>>>(input, output, N, H, W, C / 2, HO, WO);
      } else {
        dinoml::bilinear_upsampling_2d_add_kernel<
            ElemType,
            half2,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, res, output, N, H, W, C / 2, HO, WO);
      }
    } else if constexpr (std::is_same_v<ElemType, bfloat16>) {
      if (res == nullptr) {
        dinoml::bilinear_upsampling_2d_kernel<
            ElemType,
            bfloat162,
            IndexType,
            AlignCorners>
            <<<grid, block, 0, stream>>>(input, output, N, H, W, C / 2, HO, WO);
      } else {
        dinoml::bilinear_upsampling_2d_add_kernel<
            ElemType,
            bfloat162,
            IndexType,
            AlignCorners><<<grid, block, 0, stream>>>(
            input, res, output, N, H, W, C / 2, HO, WO);
      }
    } else {
      throw std::runtime_error(
          "Unsupported workload for this bilinear upsampling specialization.");
    }
  } else if (
      Mode == dinoml::Upsampling2dMode::NEAREST ||
      Mode == dinoml::Upsampling2dMode::NEAREST_EXACT) {
    if constexpr (std::is_same_v<ElemType, float>) {
      if (res == nullptr) {
        if (Alignment == 1) {
          dinoml::
              nearest_upsampling_2d_kernel<ElemType, float, IndexType, 1, Exact>
              <<<grid, block, 0, stream>>>(
                  (const float*)input, (float*)output, N, H, W, C, HO, WO);
        } else {
          dinoml::nearest_upsampling_2d_kernel<
              ElemType,
              float2,
              IndexType,
              2,
              Exact><<<grid, block, 0, stream>>>(
              (const float2*)input, (float2*)output, N, H, W, C / 2, HO, WO);
        }
      } else {
        if (Alignment == 1) {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              float,
              IndexType,
              1,
              Exact><<<grid, block, 0, stream>>>(
              (const float*)input,
              (const float*)res,
              (float*)output,
              N,
              H,
              W,
              C,
              HO,
              WO);
        } else {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              float2,
              IndexType,
              2,
              Exact><<<grid, block, 0, stream>>>(
              (const float2*)input,
              (const float2*)res,
              (float2*)output,
              N,
              H,
              W,
              C / 2,
              HO,
              WO);
        }
      }
    } else if constexpr (std::is_same_v<ElemType, half>) {
      if (res == nullptr) {
        if (Alignment == 1) {
          dinoml::
              nearest_upsampling_2d_kernel<ElemType, half, IndexType, 1, Exact>
              <<<grid, block, 0, stream>>>(
                  (const half*)input, (half*)output, N, H, W, C, HO, WO);
        } else if (Alignment == 8) {
          dinoml::nearest_upsampling_2d_kernel<
              ElemType,
              float4,
              IndexType,
              8,
              Exact><<<grid, block, 0, stream>>>(
              (const float4*)input, (float4*)output, N, H, W, C / 8, HO, WO);
        } else {
          dinoml::
              nearest_upsampling_2d_kernel<ElemType, half2, IndexType, 2, Exact>
              <<<grid, block, 0, stream>>>(
                  (const half2*)input, (half2*)output, N, H, W, C / 2, HO, WO);
        }
      } else {
        if (Alignment == 1) {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              half,
              IndexType,
              1,
              Exact><<<grid, block, 0, stream>>>(
              (const half*)input,
              (const half*)res,
              (half*)output,
              N,
              H,
              W,
              C,
              HO,
              WO);
        } else if (Alignment == 8) {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              float4,
              IndexType,
              8,
              Exact><<<grid, block, 0, stream>>>(
              (const float4*)input,
              (const float4*)res,
              (float4*)output,
              N,
              H,
              W,
              C / 8,
              HO,
              WO);
        } else {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              half2,
              IndexType,
              2,
              Exact><<<grid, block, 0, stream>>>(
              (const half2*)input,
              (const half2*)res,
              (half2*)output,
              N,
              H,
              W,
              C / 2,
              HO,
              WO);
        }
      }
    } else if constexpr (std::is_same_v<ElemType, bfloat16>) {
      if (res == nullptr) {
        if (Alignment == 1) {
          dinoml::nearest_upsampling_2d_kernel<
              ElemType,
              bfloat16,
              IndexType,
              1,
              Exact><<<grid, block, 0, stream>>>(
              (const bfloat16*)input, (bfloat16*)output, N, H, W, C, HO, WO);
        } else if (Alignment == 8) {
          dinoml::nearest_upsampling_2d_kernel<
              ElemType,
              float4,
              IndexType,
              8,
              Exact><<<grid, block, 0, stream>>>(
              (const float4*)input, (float4*)output, N, H, W, C / 8, HO, WO);
        } else {
          dinoml::nearest_upsampling_2d_kernel<
              ElemType,
              bfloat162,
              IndexType,
              2,
              Exact><<<grid, block, 0, stream>>>(
              (const bfloat162*)input,
              (bfloat162*)output,
              N,
              H,
              W,
              C / 2,
              HO,
              WO);
        }
      } else {
        if (Alignment == 1) {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              bfloat16,
              IndexType,
              1,
              Exact><<<grid, block, 0, stream>>>(
              (const bfloat16*)input,
              (const bfloat16*)res,
              (bfloat16*)output,
              N,
              H,
              W,
              C,
              HO,
              WO);
        } else if (Alignment == 8) {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              float4,
              IndexType,
              8,
              Exact><<<grid, block, 0, stream>>>(
              (const float4*)input,
              (const float4*)res,
              (float4*)output,
              N,
              H,
              W,
              C / 8,
              HO,
              WO);
        } else {
          dinoml::nearest_upsampling_2d_add_kernel<
              ElemType,
              bfloat162,
              IndexType,
              2,
              Exact><<<grid, block, 0, stream>>>(
              (const bfloat162*)input,
              (const bfloat162*)res,
              (bfloat162*)output,
              N,
              H,
              W,
              C / 2,
              HO,
              WO);
        }
      }
    } else {
      throw std::runtime_error(
          "Unsupported workload for this nearest upsampling specialization.");
    }
  } else {
    throw std::runtime_error("Unsupported upsampling mode.");
  }
}