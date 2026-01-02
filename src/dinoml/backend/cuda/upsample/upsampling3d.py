from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.backend.common import upsampling3d_common


@registry.reg("cuda.upsampling3d.gen_function")
def gen_function(
    func_attrs,
    template_path,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    return upsampling3d_common.gen_function(
        func_attrs,
        template_path,
        exec_cond_template,
        shape_eval_template,
        shape_save_template,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.upsampling3d.func_decl")
def upsampling2d_gen_function_decl(func_attrs):
    return upsampling3d_common.gen_function_decl(func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.upsampling3d.func_call")
def upsampling2d_gen_function_call(func_attrs, indent="    "):
    return upsampling3d_common.gen_function_call(
        func_attrs, backend_spec=CUDASpec(), indent=indent
    )
