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


def mk_ck_lib(scripts_path, ck_lib_path):
    with open(os.path.join(ck_lib_path, "__init__.py"), "w") as fo:
        fo.write("from . import library\n")
        fo.write("from . import generator\n")
        fo.write("from . import manifest\n")
        fo.write("from . import gemm_operation\n")
        fo.write("from . import conv2d_operation\n")

    def process_code(src_path, dst_path, code_set):
        pattern = re.compile(r"from\s([a-z_0-9]+)\simport \*")
        with open(src_path) as fi:
            lines = fi.readlines()
        output = []

        for line in lines:
            match = pattern.match(line)
            if match is not None:
                name = match.groups()[0]
                if name + ".py" in code_set:
                    line = "from .{name} import *\n".format(name=name)
                    
            output.append(line)
        with open(dst_path, "w") as fo:
            fo.writelines(output)

    srcs = os.listdir(scripts_path)
    for file in srcs:
        src_path = os.path.join(scripts_path, file)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(ck_lib_path, file)
        process_code(src_path, dst_path, srcs)
    return ck_lib_path

def main() -> None:
    scripts_path = pathlib.Path(__file__).parent.resolve()
    ck_lib_path = pathlib.Path(__file__).parent.resolve().parent.parent.joinpath("src/honey/utils/ck_lib")
    print(scripts_path)
    print(ck_lib_path)
    mk_ck_lib(
        scripts_path=scripts_path,
        ck_lib_path=ck_lib_path,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
