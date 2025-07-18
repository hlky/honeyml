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
Nhwc 3 channel to 4 channel padding.
"""

import jinja2

from honey.compiler.ops.padding.nhwc_pad_common import nhwc_pad_common


SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}HO = HI;
{{indent}}{{dtype}}WO = WI;
{{indent}}{{dtype}}CO = 4;
"""
)


class nhwc3to4(nhwc_pad_common):
    def __init__(self, shape_func_template=None, padded_channels=None):
        super().__init__(
            SHAPE_FUNC_TEMPLATE if shape_func_template is None else shape_func_template,
            4 if padded_channels is None else padded_channels,
        )
