from typing import Any, Dict, Tuple

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/upsampling_3d.h>
#include <stdexcept>
#include <algorithm>

void {{function_name}} (
    const void* in_ptr,
    const void* res_ptr,
    void* out_ptr,
    {{index_type}}* batch,
    {{index_type}}* in_f,
    {{index_type}}* in_h,
    {{index_type}}* in_w,
    {{index_type}}* in_ch,
    {{index_type}}* out_batch,
    {{index_type}}* out_f,
    {{index_type}}* out_h,
    {{index_type}}* out_w,
    dinoml::DeviceStream stream
) {
  // Read runtime shapes (NFHWC)
  const {{index_type}} N  = *batch;
  const {{index_type}} F  = *in_f;
  const {{index_type}} H  = *in_h;
  const {{index_type}} W  = *in_w;
  const {{index_type}} C  = *in_ch;

  // We implement scale_factor=2.0 behavior
  const {{index_type}} HO = H * 2;
  const {{index_type}} WO = W * 2;

  // Output frames follow the PyTorch branching you showed:
  // if F>1 and odd: FO = 2*F - 1 (first frame 2D, rest 3D)
  // else if F>1:    FO = 2*F
  // else:           FO = 1
  {{index_type}} FO;
  if (F > 1 && (F & 1)) {
    FO = 2 * F - 1;
  } else if (F > 1) {
    FO = 2 * F;
  } else {
    FO = 1;
  }

  *out_batch = N;
  *out_f = FO;
  *out_h = HO;
  *out_w = WO;

  const {{dtype}}* in_t  = static_cast<const {{dtype}}*>(in_ptr);
  {{dtype}}* out_t       = static_cast<{{dtype}}*>(out_ptr);

  // NOTE: pointer offsets are in ElemType units (not VectorType units)
  const int64_t in_frame_stride  = (int64_t)C * (int64_t)H  * (int64_t)W;
  const int64_t out_frame_stride = (int64_t)C * (int64_t)HO * (int64_t)WO;

  if (F == 1) {
    // Treat as 3D with F=1 -> FO=1
    upsampling_3d_launcher<
        {{dtype}},
        {{vec_type}},
        int64_t,
        {{alignment}},
        {{mode}},
        {{align_corners}},
        {{exact}},
        {{has_residual}}>
    (
      in_t,
      static_cast<const {{dtype}}*>(res_ptr),
      out_t,
      N,
      /*FI*/ 1,
      H,
      W,
      C,
      /*FO*/ 1,
      HO,
      WO,
      stream
    );
    return;
  }

  if ((F & 1) == 0) {
    // Even frames: one 3D upsample launch, FO = 2F
    upsampling_3d_launcher<
        {{dtype}},
        {{vec_type}},
        int64_t,
        {{alignment}},
        {{mode}},
        {{align_corners}},
        {{exact}},
        {{has_residual}}>
    (
      in_t,
      static_cast<const {{dtype}}*>(res_ptr),
      out_t,
      N,
      /*FI*/ F,
      H,
      W,
      C,
      /*FO*/ ({{index_type}})(2 * F),
      HO,
      WO,
      stream
    );
    return;
  }

  // Odd frames: two launches, no slice/concat
  // A) first frame only: FI=1 -> FO=1
  upsampling_3d_launcher<
      {{dtype}},
      {{vec_type}},
      int64_t,
      {{alignment}},
      {{mode}},
      {{align_corners}},
      {{exact}},
      {{has_residual}}>
  (
    in_t,
    static_cast<const {{dtype}}*>(res_ptr),
    out_t,
    N,
    /*FI*/ 1,
    H,
    W,
    C,
    /*FO*/ 1,
    HO,
    WO,
    stream
  );

  // B) rest frames: input offset by 1 frame, output offset by 1 output-frame
  //    FI = F-1, FO = 2*(F-1)
  const {{dtype}}* in_rest  = in_t  + in_frame_stride;
  {{dtype}}* out_rest       = out_t + out_frame_stride;

  upsampling_3d_launcher<
      {{dtype}},
      {{vec_type}},
      int64_t,
      {{alignment}},
      {{mode}},
      {{align_corners}},
      {{exact}},
      {{has_residual}}>
  (
    in_rest,
    static_cast<const {{dtype}}*>(res_ptr), // residual (if any) is expected to match output; if you use residual here, you likely need matching offsets too
    out_rest,
    N,
    /*FI*/ ({{index_type}})(F - 1),
    H,
    W,
    C,
    /*FO*/ ({{index_type}})(2 * (F - 1)),
    HO,
    WO,
    stream
  );

  return;
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    r"""
void {{func_name}}(
  const void*,
  const void*,
  void*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{res_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_f}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_f}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    stream
{{indent}});
"""
)


def _gen_alignment(x) -> int:
    in_channel = x.shape()[-1].value()
    if in_channel % 8 == 0:
        return 8
    if in_channel % 4 == 0:
        return 4
    if in_channel % 2 == 0:
        return 2
    return 1


def _pick_vec_type(input_type: str, alignment: int, mode: str) -> Tuple[str, int]:
    if mode == "trilinear":
        # force vector2
        if alignment > 1:
            alignment = 2
        if input_type == "float":
            return "float2", alignment
        if input_type == "bfloat16":
            return "bfloat162", alignment
        return "half2", alignment

    # nearest / nearest-exact
    if input_type == "float":
        if alignment > 1:
            return "float2", 2
        return "float", 1

    if input_type == "half":
        if alignment >= 8:
            return "float4", 8
        if alignment >= 2:
            return "half2", 2
        return "half", 1

    if input_type == "bfloat16":
        if alignment >= 8:
            return "float4", 8
        if alignment >= 2:
            return "bfloat162", 2
        return "bfloat16", 1

    raise RuntimeError(f"Unsupported dtype: {input_type}")


def gen_function(
    func_attrs: Dict[str, Any], backend_spec: CUDASpec, bias_add: bool = False
) -> str:
    func_name = func_attrs["name"]
    x = func_attrs["inputs"][0]

    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    alignment = _gen_alignment(x)

    mode_map = {
        "trilinear": "dinoml::Upsampling3dMode::TRILINEAR",
        "nearest": "dinoml::Upsampling3dMode::NEAREST",
        "nearest-exact": "dinoml::Upsampling3dMode::NEAREST_EXACT",
    }
    mode = mode_map[func_attrs["mode"]]

    align_corners = (
        "false"
        if func_attrs.get("align_corners") is None
        else str(func_attrs["align_corners"]).lower()
    )
    exact = str(func_attrs["mode"] == "nearest-exact").lower()
    has_residual = str(bias_add).lower()

    vec_type, alignment = _pick_vec_type(input_type, alignment, func_attrs["mode"])

    return SRC_TEMPLATE.render(
        function_name=func_name,
        dtype=input_type,
        vec_type=vec_type,
        alignment=alignment,
        mode=mode,
        align_corners=align_corners,
        exact=exact,
        has_residual=has_residual,
        index_type=backend_spec.index_type,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec: CUDASpec) -> str:
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
    )


def gen_function_call(
    func_attrs: Dict[str, Any],
    backend_spec: CUDASpec,
    indent="  ",
    bias_add: bool = False,
) -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    xshape = x._attrs["shape"]
    yshape = y._attrs["shape"]

    res_ptr = "nullptr"
    if bias_add:
        res_ptr = func_attrs["inputs"][1]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        res_ptr=res_ptr,
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_f="&" + xshape[1]._attrs["name"],
        p_in_h="&" + xshape[2]._attrs["name"],
        p_in_w="&" + xshape[3]._attrs["name"],
        p_in_ch="&" + xshape[4]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_f="&" + yshape[1]._attrs["name"],
        p_out_h="&" + yshape[2]._attrs["name"],
        p_out_w="&" + yshape[3]._attrs["name"],
        indent=indent,
    )


@registry.reg("cuda.upsampling3d_compress_time.gen_function")
def cuda_upsampling3d_compress_time_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec(), bias_add=False)


@registry.reg("cuda.upsampling3d_compress_time.func_decl")
def cuda_upsampling3d_compress_time_func_decl(func_attrs):
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.upsampling3d_compress_time.func_call")
def cuda_upsampling3d_compress_time_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, CUDASpec(), indent=indent, bias_add=False)
