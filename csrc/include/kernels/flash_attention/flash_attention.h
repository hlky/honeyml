#pragma once

#include <stdexcept>

#include <flash_attn/flash_attn_dinoml.h>

#include <dinoml/device.h>
#include <dinoml/helpers.h>
#include "mask_type.h"
#include "model_interface.h"

namespace dinoml {

inline flash::MaskType ConvertMaskType(DinoMLMaskType mask_type) {
  switch (mask_type) {
    case DinoMLMaskType::kNone:
      return flash::MaskType::kNone;
    case DinoMLMaskType::kCausalFromTopLeft:
      return flash::MaskType::kCausalFromTopLeft;
    case DinoMLMaskType::kCausalFromBottomRight:
      return flash::MaskType::kCausalFromBottomRight;
    default:
      throw std::runtime_error("Unhandled MaskType");
  }
}

inline flash::DataType ConvertDtype(DinoMLDtype dtype) {
  switch (dtype) {
    case DinoMLDtype::kHalf:
      return flash::DataType::kFloat16;
    case DinoMLDtype::kBFloat16:
      return flash::DataType::kBFloat16;
    default:
      throw std::runtime_error("Unhandled dtype");
  }
}

}

inline void FlashAttention(
    void* output,
    int64_t output_batch_stride,
    int64_t output_row_stride,
    int64_t output_head_stride,
    void* q,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    void* k,
    int64_t k_batch_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    void* v,
    int64_t v_batch_stride,
    int64_t v_row_stride,
    int64_t v_head_stride,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t seqlen_k,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    dinoml::DinoMLMaskType mask_type,
    DinoMLDtype dtype,
    int window_size_left,
    int window_size_right,
    void* workspace,
    dinoml::DeviceStream stream) {
  auto flash_mask_type = dinoml::ConvertMaskType(mask_type);

  // TODO
  int32_t num_splits = 1;

  void* softmax_lse_ptr = workspace;

  char* workspace_ptr =
      static_cast<char*>(workspace) +
      dinoml::helpers::round_up_to_multiple(
          batch_size * num_heads_q * seqlen_q * sizeof(float), 16);

  flash::FlashAttentionLauncher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      q,
      q_batch_stride,
      q_row_stride,
      q_head_stride,
      k,
      k_batch_stride,
      k_row_stride,
      k_head_stride,
      v,
      v_batch_stride,
      v_row_stride,
      v_head_stride,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      flash_mask_type,
      workspace,
      dinoml::ConvertDtype(dtype),
      window_size_left,
      window_size_right,
      num_splits,
      nullptr,
      nullptr,
      stream);
}
