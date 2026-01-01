#  Copyright 2025 hlky. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Codegen functions for upsampling2d.
"""

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.backend.common import upsampling2d_common


@registry.reg("cuda.upsampling2d.gen_function")
def gen_function(
    func_attrs,
    template_path,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    return upsampling2d_common.gen_function(
        func_attrs,
        template_path,
        exec_cond_template,
        shape_eval_template,
        shape_save_template,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.upsampling2d.func_decl")
def upsampling2d_gen_function_decl(func_attrs):
    return upsampling2d_common.gen_function_decl(func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.upsampling2d.func_call")
def upsampling2d_gen_function_call(func_attrs, indent="    "):
    return upsampling2d_common.gen_function_call(
        func_attrs, backend_spec=CUDASpec(), indent=indent
    )
