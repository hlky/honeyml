from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    """
#include <dinoml/device.h>
#include <dinoml/helpers.h>

#include <kernels/flash_attention/flash_attention.h>

void {{function_name}}(
    void* out,
    const void* q,
    const void* k,
    const void* v,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t seqlen_k,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    void* workspace,
    dinoml::DeviceStream stream
) {
  const int64_t q_head_stride = head_dim;
  const int64_t q_row_stride  = num_heads_q * head_dim;
  const int64_t q_batch_stride = seqlen_q * q_row_stride;

  const int64_t k_head_stride = head_dim;
  const int64_t k_row_stride  = num_heads_k * head_dim;
  const int64_t k_batch_stride = seqlen_k * k_row_stride;

  const int64_t v_head_stride = head_dim;
  const int64_t v_row_stride  = num_heads_k * head_dim;
  const int64_t v_batch_stride = seqlen_k * v_row_stride;

  const int64_t out_head_stride = head_dim;
  const int64_t out_row_stride  = num_heads_q * head_dim;
  const int64_t out_batch_stride = seqlen_q * out_row_stride;

  FlashAttention(
      out,
      out_batch_stride,
      out_row_stride,
      out_head_stride,
      const_cast<void*>(q),
      q_batch_stride,
      q_row_stride,
      q_head_stride,
      const_cast<void*>(k),
      k_batch_stride,
      k_row_stride,
      k_head_stride,
      const_cast<void*>(v),
      v_batch_stride,
      v_row_stride,
      v_head_stride,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      {{mask_type}},
      {{dtype}},
      {{window_size_left}},
      {{window_size_right}},
      workspace,
      stream
  );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void*,
    const void*,
    const void*,
    const void*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    void*,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{out}},
{{indent}}    {{q}},
{{indent}}    {{k}},
{{indent}}    {{v}},
{{indent}}    {{batch_size}},
{{indent}}    {{seqlen_q}},
{{indent}}    {{seqlen_k}},
{{indent}}    {{num_heads_q}},
{{indent}}    {{num_heads_k}},
{{indent}}    {{head_dim}},
{{indent}}    global_workspace_,
{{indent}}    stream
{{indent}});
"""
)


def _dim_to_str(d) -> str:
    if isinstance(d, IntImm):
        return str(d._attrs["values"][0])
    if isinstance(d, IntVar):
        return d._attrs["name"]
    raise RuntimeError(f"Expected IntImm/IntVar, got: {type(d)}")


def gen_function(func_attrs: Dict[str, Any], backend_spec: CUDASpec) -> str:
    func_name = func_attrs["name"]

    q = func_attrs["inputs"][0]
    dtype = q._attrs["dtype"]
    dtype_map = {
        "float16": "DinoMLDtype::kHalf",
        "bfloat16": "DinoMLDtype::kBFloat16",
    }
    dtype = dtype_map.get(dtype)

    mask_type = func_attrs.get("mask_type", "none")
    mask_map = {
        "none": "dinoml::DinoMLMaskType::kNone",
        "causal_topleft": "dinoml::DinoMLMaskType::kCausalFromTopLeft",
        "causal_bottomright": "dinoml::DinoMLMaskType::kCausalFromBottomRight",
    }
    mask_type = mask_map.get(mask_type, "dinoml::DinoMLMaskType::kNone")

    return SRC_TEMPLATE.render(
        function_name=func_name,
        dtype=dtype,
        mask_type=mask_type,
        window_size_left=int(func_attrs.get("window_size_left", -1)),
        window_size_right=int(func_attrs.get("window_size_right", -1)),
        num_splits=int(func_attrs.get("num_splits", 0)),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec: CUDASpec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


def gen_function_call(func_attrs: Dict[str, Any], indent: str = "  ") -> str:
    q, k, v = func_attrs["inputs"]
    out = func_attrs["outputs"][0]

    b = _dim_to_str(q._attrs["shape"][0])
    seqlen_q = _dim_to_str(q._attrs["shape"][1])
    num_heads_q = _dim_to_str(q._attrs["shape"][2])
    head_dim = _dim_to_str(q._attrs["shape"][3])

    seqlen_k = _dim_to_str(k._attrs["shape"][1])
    num_heads_k = _dim_to_str(k._attrs["shape"][2])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out=out._attrs["name"],
        q=q._attrs["name"],
        k=k._attrs["name"],
        v=v._attrs["name"],
        batch_size=b,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        indent=indent,
    )


@registry.reg("cuda.flash_attn.gen_function")
def cuda_flash_attention_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.flash_attn.func_decl")
def cuda_flash_attention_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.flash_attn.func_call")
def cuda_flash_attention_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
