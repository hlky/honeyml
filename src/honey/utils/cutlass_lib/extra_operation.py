
import enum
import os.path
import shutil
import argparse

from .library import *
from .manifest import *
from .generator import CreateGemmOperator


def perm021_fc(manifest, args):

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                        [16, 8, 16],                                          DataType.f16, DataType.f16, DataType.f32,             OpcodeClass.TensorOp,                                 MathOperation.multiply_add),
    MathInstruction(                                        [16, 8, 16],                                          DataType.f16, DataType.f16, DataType.f16,             OpcodeClass.TensorOp,                                 MathOperation.multiply_add),
    MathInstruction(                                        [16, 8, 16],                                          DataType.bf16, DataType.bf16, DataType.f32,           OpcodeClass.TensorOp,                                 MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024
  max_cc_smem_limited = 80

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  32, 32],  7, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions,       data_type, alignment_constraints)

    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions,         data_type_mixed, alignment_constraints)


def GenerateSM80(manifest, args):
    perm021_fc(manifest, args)