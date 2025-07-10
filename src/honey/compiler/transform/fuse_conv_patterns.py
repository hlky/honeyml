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
from functools import partial
from honey.compiler.ops.common import elementwise
from honey.compiler.ops.common.epilogue import FuncEnum
from honey.compiler.ops.conv import (
    conv2d,
    transposed_conv2d,
)


def get_conv2d_bias_pattern():
    # Attribute in conv2d is not of concern, it will be passed-through directly.
    return [((conv2d(stride=1, pad=0, bias=False), elementwise(FuncEnum.ADD)), partial(conv2d(bias=True)),)]


def get_conv2d_bias_elementwise_patterns():
    """
    We create the pattern of fusion here.
    The format should be in the form of (pattern, replacement)

    pattern: This would be a list of operator which are chained which we
             want to match
    replacement: The op to replace pattern.
    """

    conv2d_bias_patterns = [
        (
            (
                conv2d(stride=1, pad=0, bias=True),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.RELU),
            ),
            partial(conv2d(activation="relu", add=True, bias=True)),
        ),
        (
            (
                conv2d(stride=1, pad=0, bias=True),
                elementwise(FuncEnum.RELU),
            ),
            partial(conv2d(activation="relu", bias=True)),
        ),
        (
            (
                conv2d(stride=1, pad=0, bias=True),
                elementwise(FuncEnum.SIGMOID),
            ),
            partial(conv2d(activation="sigmoid", add=True, bias=True)),
        ),
    ]

    transposed_conv2d_bias_patterns = [
        (
            (
                transposed_conv2d(stride=1, pad=0, bias=True),
                elementwise(FuncEnum.RELU),
            ),
            partial(transposed_conv2d(bias=True, activation="relu")),
        ),
    ]

    transposed_conv2d_patterns = [
        (
            (
                transposed_conv2d(stride=1, pad=0, bias=False),
                elementwise(FuncEnum.ADD),
                elementwise(FuncEnum.RELU),
            ),
            partial(transposed_conv2d(bias=True, activation="relu")),
        ),
        (
            (
                transposed_conv2d(stride=1, pad=0, bias=True),
                elementwise(FuncEnum.RELU),
            ),
            partial(transposed_conv2d(bias=True, activation="relu")),
        ),
    ]

    fusion_patterns = (
        conv2d_bias_patterns
        + transposed_conv2d_bias_patterns
        + transposed_conv2d_patterns
    )

    return fusion_patterns


def get_cuda_only_conv2d_bias_elementwise_patterns():
    conv2d_bias_patterns = [
        (
            (
                conv2d(stride=1, pad=0, bias=True, few_channels=True),
                elementwise(FuncEnum.RELU),
            ),
            partial(conv2d(bias=True, few_channels=True, activation="relu")),
        ),
        (
            (
                conv2d(stride=1, pad=0, bias=True),
                elementwise(FuncEnum.ADD),
            ),
            partial(conv2d(add=True, bias=True)),
        ),
    ]

    transposed_conv2d_patterns = [
        (
            (
                transposed_conv2d(stride=1, pad=0, bias=False),
                elementwise(FuncEnum.ADD),
            ),
            partial(transposed_conv2d(bias=True)),
        ),
    ]

    return conv2d_bias_patterns + transposed_conv2d_patterns
