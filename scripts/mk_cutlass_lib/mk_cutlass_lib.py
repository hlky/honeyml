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
import os
import pathlib
import re

import extra_conv_emit
import extra_cutlass_generator
import extra_enum
import extra_gemm_emit


def mk_cutlass_lib(scripts_path, cutlass_lib_path):
    with open(os.path.join(cutlass_lib_path, "__init__.py"), "w") as fo:
        fo.write("from . import library\n")
        fo.write("from . import generator\n")
        fo.write("from . import manifest\n")
        fo.write("from . import conv3d_operation\n")
        fo.write("from . import gemm_operation\n")
        fo.write("from . import conv2d_operation\n")
        fo.write("from . import extra_operation\n")

    def process_code(src_path, dst_path, code_set):
        pattern = re.compile(r"from\s([a-z_0-9]+)\simport\s(.+)")
        with open(src_path, newline="\n") as fi:
            lines = fi.readlines()
        output = []

        for line in lines:
            match = pattern.match(line)
            if match is not None:
                name, _import = match.groups()
                if name + ".py" in code_set:
                    line = "from .{name} import {_import}\n".format(name=name, _import=_import)
            output.append(line)
        if "library.py" in dst_path:
            lines = extra_enum.emit_library()
            output.append(lines)
        if "conv2d_operation.py" in dst_path:
            lines = extra_conv_emit.emit_library()
            output.append(lines)
        if "gemm_operation.py" in dst_path:
            lines = extra_gemm_emit.emit_library()
            output.append(lines)
        with open(dst_path, "w") as fo:
            fo.writelines(output)

    srcs = os.listdir(scripts_path)
    if "__init__.py" in srcs:
        srcs.remove("__init__.py")
    for file in srcs:
        src_path = os.path.join(scripts_path, file)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(cutlass_lib_path, file)
        process_code(src_path, dst_path, srcs)

    # extra configs
    dst_path = os.path.join(cutlass_lib_path, "extra_operation.py")
    with open(dst_path, "w", newline="\n") as fo:
        code = extra_cutlass_generator.emit_library()
        fo.write(code)
    return cutlass_lib_path


def main() -> None:
    scripts_path = pathlib.Path(__file__).parent.resolve().parent.parent.joinpath("3rdparty/cutlass/tools/library/scripts")
    cutlass_lib_path = pathlib.Path(__file__).parent.resolve().parent.parent.joinpath("src/dinoml/utils/cutlass_lib")
    print(scripts_path)
    print(cutlass_lib_path)
    mk_cutlass_lib(
        scripts_path=scripts_path,
        cutlass_lib_path=cutlass_lib_path,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
