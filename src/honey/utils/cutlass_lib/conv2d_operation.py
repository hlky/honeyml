#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#
#

import enum
import os.path
import shutil

from .library import *

###################################################################################################

#
class Conv2dOperation:
  #
  def __init__(self, conv_kind, iterator_algorithm, arch, tile_description, A, B, C, element_epilogue, \
    stride_support, epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity1, \
    group_mode = GroupMode.NoneGroup):

    self.operation_kind = OperationKind.Conv2d
    self.arch = arch
    self.tile_description = tile_description
    self.conv_kind = conv_kind
    self.A = A
    self.B = B
    self.C = C
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor
    self.iterator_algorithm = iterator_algorithm
    self.stride_support = stride_support
    self.swizzling_functor = swizzling_functor
    self.group_mode = group_mode
  #
  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian
      ]
    return self.tile_description.math_instruction.math_operation in complex_operators
  
  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator

    if self.is_complex():
      return get_complex_from_real(accum)

    return accum

  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    intermediate_type = ''

    if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp:
      inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.accumulator_type():
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]
    else:
      inst_shape = ''

    return "%s%s%s%s_%s" % (ShortDataTypeNames[self.accumulator_type()], \
      inst_shape, intermediate_type, ConvKindNames[self.conv_kind], IteratorAlgorithmNames[self.iterator_algorithm])

  #
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    if self.C.element != self.tile_description.math_instruction.element_accumulator and \
      self.A.element != self.tile_description.math_instruction.element_accumulator:
      extended_name = "${element_c}_${core_name}_${element_a}"
    elif self.C.element == self.tile_description.math_instruction.element_accumulator and  \
      self.A.element != self.tile_description.math_instruction.element_accumulator:
      extended_name = "${core_name}_${element_a}"
    else:
      extended_name = "${core_name}"

    extended_name = SubstituteTemplate(extended_name, {
      'element_a': DataTypeNames[self.A.element],
      'element_c': DataTypeNames[self.C.element],
      'core_name': self.core_name()
      })

    return extended_name

  #
  def layout_name(self):
    return "%s" % (ShortLayoutTypeNames[self.A.layout])

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''

    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    
    threadblock = self.tile_description.procedural_name()

    # grouped conv
    if self.group_mode != GroupMode.NoneGroup:
      group_conv_name = f"{GroupModeNames[self.group_mode]}_"
    else:
      group_conv_name = ""

    if self.stride_support == StrideSupport.Unity:
      configuration_name = "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_unity_stride_${group_conv_name}align${alignment}"
    else:
      configuration_name = "cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_${group_conv_name}align${alignment}"

    return SubstituteTemplate(
      configuration_name,
      {
        'opcode_class': opcode_class_name,
        'extended_name': self.extended_name(),
        'threadblock': threadblock,
        'layout': self.layout_name(),
        'alignment': "%d" % self.A.alignment,
        'group_conv_name': group_conv_name
      }
    )

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.configuration_name()

###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

class EmitConv2dInstance:
  def __init__(self):
    self.template = """
  // Conv2d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
  using ${operation_name}_base = 
  typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}<
    ${element_a}, 
    ${layout_a},
    ${element_b}, 
    ${layout_b},
    ${element_c}, 
    ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator},
    ${iterator_algorithm},
    ${stride_support},
    ${align_a},
    ${align_b}
  >::Kernel;
"""
    self.template_group_conv = """
  // Conv2d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
  using ${operation_name}_base =
  typename cutlass::conv::kernel::DefaultConv2dGroup${conv_kind_name}<
    ${element_a},
    ${layout_a},
    ${element_b},
    ${layout_b},
    ${element_c},
    ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator},
    ${group_mode},
    ${iterator_algorithm},
    ${stride_support},
    ${align_a},
    ${align_b}
  >::Kernel;
"""
    self.template_depthwise_direct_conv = """
  // Conv2d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
  using ${operation_name}_base =
  typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConv${conv_kind_name}<
    ${element_a},
    ${layout_a},
    ${element_b},
    ${layout_b},
    ${element_c},
    ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::conv::TensorNHWCShape<${threadblock_output_shape_n}, ${threadblock_output_shape_p}, ${threadblock_output_shape_q}, ${groups_per_cta}>,
    cutlass::MatrixShape<${filter_shape_r}, ${filter_shape_s}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue},
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >,

    cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ${threadblock_output_shape_n},
          ${threadblock_output_shape_p},
          ${threadblock_output_shape_q}>, 
    ${stages},
    ${math_operator},
    ${iterator_algorithm},
    ${stride_support},
    cutlass::MatrixShape<${stride_r}, ${stride_s}>,
    cutlass::MatrixShape<${dilation_r}, ${dilation_s}>
  >::Kernel;
"""

  def emit(self, operation):

    warp_shape = [int(operation.tile_description.threadblock_shape[idx] / operation.tile_description.warp_count[idx]) for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
      'conv_kind': ConvKindTag[operation.conv_kind],
      'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()], 
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'iterator_algorithm': IteratorAlgorithmTag[operation.iterator_algorithm],
      'iterator_algorithm_name': IteratorAlgorithmNames[operation.iterator_algorithm].capitalize(),
      'stride_support': StrideSupportTag[operation.stride_support],
      'math_operator': 'cutlass::arch::OpMultiplyAddComplex' if operation.is_complex() else \
      MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
    }

    if operation.group_mode == GroupMode.NoneGroup:
      return SubstituteTemplate(self.template, values)

    elif operation.group_mode == GroupMode.Depthwise:
      values['group_mode'] = GroupModeTag[operation.group_mode]
      # Setup other template params
      values['threadblock_output_shape_n'] = str(operation.tile_description.threadblock_output_shape[0])
      values['threadblock_output_shape_p'] = str(operation.tile_description.threadblock_output_shape[1])
      values['threadblock_output_shape_q'] = str(operation.tile_description.threadblock_output_shape[2])
      
      values['groups_per_cta'] = str(operation.tile_description.threadblock_output_shape[3])

      values['filter_shape_r'] = str(operation.tile_description.filter_shape[0])
      values['filter_shape_s'] = str(operation.tile_description.filter_shape[1])

      values['stride_r'] = str(operation.tile_description.stride[0])
      values['stride_s'] = str(operation.tile_description.stride[1])

      values['dilation_r'] = str(operation.tile_description.dilation[0])
      values['dilation_s'] = str(operation.tile_description.dilation[1])

      return SubstituteTemplate(self.template_depthwise_direct_conv, values)

    else:
      values['group_mode'] = GroupModeTag[operation.group_mode]
      return SubstituteTemplate(self.template_group_conv, values)

###################################################################################################
#
# Generator functions for all layouts
#
###################################################################################################

#
def GenerateConv2dTensorOp(manifest, tile_descriptions, min_cc, align = 128):

  for tile in tile_descriptions:
    for conv_kind in [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad]:

      if conv_kind == ConvKind.Fprop or (tile.math_instruction.element_accumulator in [DataType.f16, DataType.f32]):

        #
        output_types = [tile.math_instruction.element_a, tile.math_instruction.element_accumulator] \
          if DataTypeSize[tile.math_instruction.element_accumulator] == 32 \
          else [tile.math_instruction.element_accumulator,]

        for output_type in output_types:
          A = TensorDescription(tile.math_instruction.element_a, LayoutType.TensorNHWC, int(align / DataTypeSize[tile.math_instruction.element_a]))
          B = TensorDescription(tile.math_instruction.element_b, LayoutType.TensorNHWC, int(align / DataTypeSize[tile.math_instruction.element_b]))
          C = TensorDescription(output_type,  LayoutType.TensorNHWC, max(1, int(align / DataTypeSize[output_type])))

          manifest.append(Conv2dOperation(conv_kind, min_cc, tile, A, B, C, tile.math_instruction.element_accumulator))

###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################

class EmitConv2dConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name)

    self.instance_emitter = EmitConv2dInstance()

    self.instance_template = """
${operation_instance}

// Derived class
struct ${operation_name} : 
  public ${operation_name}_base { };

///////////////////////////////////////////////////////////////////////////////////////////////////

"""
    self.header_template = """
/*
  Generated by conv2d_operation.py - Do not edit.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "library_internal.h"
#include "conv2d_operation.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
"""

    self.configuration_header = """

namespace cutlass {
namespace library {

// Initialize all instances
void initialize_${configuration_name}(Manifest &manifest) {

"""

    self.configuration_instance = """
  using Operation_${operation_name} = cutlass::conv::device::ImplicitGemmConvolution<
    ${operation_name}>;

  manifest.append(new cutlass::library::Conv2dOperation<
    Operation_${operation_name}>(
      "${operation_name}"));

"""

    self.configuration_direct_conv_instance = """
  using Operation_${operation_name} = cutlass::conv::device::DirectConvolution<
    ${operation_name}>;

  manifest.append(new cutlass::library::DirectConv2dOperation<
    Operation_${operation_name}>(
      "${operation_name}"));

"""

    self.configuration_epilogue = """
}
"""
    self.epilogue_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  #
  def __enter__(self):
    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(SubstituteTemplate(self.header_template, {
      'configuration_name': self.configuration_name
      }))
    self.operations = []
    return self

  #
  def emit(self, operation):
    self.operations.append(operation)
    self.configuration_file.write(SubstituteTemplate(self.instance_template, {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'operation_instance': self.instance_emitter.emit(operation)
      }))

  #
  def __exit__(self, exception_type, exception_value, traceback):

    self.configuration_file.write(SubstituteTemplate(self.configuration_header, {
      'configuration_name': self.configuration_name
      }))

    for operation in self.operations:
      if operation.group_mode == GroupMode.Depthwise:
        self.configuration_file.write(SubstituteTemplate(self.configuration_direct_conv_instance, {
          'configuration_name': self.configuration_name,
          'operation_name': operation.procedural_name()  
        }))
      else: 
        self.configuration_file.write(SubstituteTemplate(self.configuration_instance, {
          'configuration_name': self.configuration_name,
          'operation_name': operation.procedural_name()  
        }))

    self.configuration_file.write(self.configuration_epilogue)
    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()


###################################################################################################
###################################################################################################

class EmitConv2dWithBroadcastInstance:
  def __init__(self):
    self.template = """
    
  using ${operation_name}_base =
  typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}WithBroadcast<
    ${element_a},
    ${layout_a},
    ${element_b},
    ${layout_b},
    ${element_c},
    ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m},
                             ${threadblock_shape_n},
                             ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m},
                             ${warp_shape_n},
                             ${warp_shape_k} >,
    cutlass::gemm::GemmShape<${instruction_shape_m},
                             ${instruction_shape_n},
                             ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${element_accumulator},
      ${element_epilogue},
      ${element_c},
      ${epilogue_vector_length},
      ${activation_op},
      ${binary_op},
      ${unary_op}
    >,
    ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator},
    ${iterator_algorithm},
    ${stride_support},
    ${align_a},
    ${align_b}
  >::Kernel;
  """

  def emit(self, operation):

    warp_shape = [int(operation.tile_description.threadblock_shape[idx]
                    / operation.tile_description.warp_count[idx]) for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
                                   / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
      'conv_kind': ConvKindTag[operation.conv_kind],
      'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'iterator_algorithm': IteratorAlgorithmTag[operation.iterator_algorithm],
      'iterator_algorithm_name': IteratorAlgorithmNames[operation.iterator_algorithm].capitalize(),
      'stride_support': StrideSupportTag[operation.stride_support],
      'math_operator': 'cutlass::arch::OpMultiplyAddComplex' if operation.is_complex() else       MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'activation_op': EpilogueMathTag[operation.activation_op],
      'binary_op': EpilogueMathTag[operation.binary_op],
      'unary_op': EpilogueMathTag[operation.unary_op],
    }

    return SubstituteTemplate(self.template, values)

