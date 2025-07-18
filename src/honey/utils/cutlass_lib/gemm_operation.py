#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import os.path
import shutil
import functools
import operator
import collections

from .library import *

###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################

#
class GemmOperation:
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, D = None, 
      kernel_schedule = KernelScheduleType.ScheduleAuto, epilogue_schedule = EpilogueScheduleType.ScheduleAuto):

    self.prefix = "3x" if gemm_kind == GemmKind.Universal3x else ""
    self.operation_kind = OperationKind.Gemm
    self.arch = arch
    self.tile_description = tile_description
    self.gemm_kind = gemm_kind
    self.A = A
    self.B = B
    self.C = C
    self.D = D 
    if self.D == None:
      self.D = self.C

    if gemm_kind != GemmKind.Universal3x:
      assert(kernel_schedule == KernelScheduleType.ScheduleAuto)
      assert(epilogue_schedule == EpilogueScheduleType.ScheduleAuto)
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor
    self.swizzling_functor = swizzling_functor

  #
  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex, 
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  #
  def is_planar_complex(self):
    return self.gemm_kind in (GemmKind.PlanarComplex, GemmKind.PlanarComplexArray)

  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator

    if self.is_complex():
      return get_complex_from_real(accum)

    return accum

  #
  def short_math_name(self):
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      return "g%s" % ShortDataTypeNames[self.accumulator_type()]
    return ShortDataTypeNames[self.accumulator_type()]


  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''
    
    inst_shape = ''
    inst_operation = ''
    intermediate_type = ''

    math_operations_map = {
      MathOperation.xor_popc: 'xor',
    }

    if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp or \
      self.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp:

      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ''

      if self.gemm_kind == GemmKind.Universal3x:
        inst_shape = "{0}x{1}x{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape))
      else:
        inst_shape = "{0}{1}{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape))

      inst_shape += math_op_string

      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

    return "%s%s%s%s" % (self.short_math_name(), inst_shape, intermediate_type, GemmKindNames[self.gemm_kind])

  # Generates a string representing the MMA instruction.
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    if self.is_complex():
      extended_name = "${core_name}"
    else:
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

  def extended_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    extended_name = "{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element],
      core_name = self.core_name())
    return extended_name

  # Generates a short string representing the AB layout tags (e.g. nt or tn)
  def layout_name(self):
    if self.is_complex() or self.is_planar_complex():
      return "%s%s" % (
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)], 
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)]
      )
    return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

  # Generates a short string representing the ABC layout tags (e.g. ntn or tnn)
  def layout_name_3x(self):
    if self.is_complex() or self.is_planar_complex():
      return "{}{}{}".format(
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)], 
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)],
        ShortComplexLayoutNames[(self.C.layout, self.C.complex_transform)])
    else:
      return "{}{}{}".format(
        ShortLayoutTypeNames[self.A.layout],
        ShortLayoutTypeNames[self.B.layout],
        ShortLayoutTypeNames[self.C.layout])

  # Generates a short string representing underlying kernel schedule type
  def kernel_schedule_name_3x(self):
    return KernelScheduleSuffixes[self.kernel_schedule]

  # Generates a short string representing underlying epilogue schedule type
  def epilogue_schedule_name_3x(self):
    return EpilogueScheduleSuffixes[self.epilogue_schedule]

  # Generates the full kernel function name
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    if self.arch >= 90:
      kernel_name_template = "cutlass{p}_sm{ar}_{op}_{ex}_{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{l}_{s}_align{al}{k}{e}"
      return kernel_name_template.format(
          p = self.prefix,
          ar = self.arch,
          op = opcode_class_name,
          ex = self.extended_name_3x(),
          tbm = self.tile_description.tile_shape[0],
          tbn = self.tile_description.tile_shape[1],
          tbk = self.tile_description.tile_shape[2],
          cm = self.tile_description.cluster_shape[0],
          cn = self.tile_description.cluster_shape[1],
          ck = self.tile_description.cluster_shape[2],
          l = self.tile_description.stages,
          s = self.layout_name_3x(),
          al = str(max(self.A.alignment, self.B.alignment)),
          k = self.kernel_schedule_name_3x(),
          e = self.epilogue_schedule_name_3x())
    else:
      threadblock = self.tile_description.procedural_name()
      return "cutlass{p}_{op}_{ex}_{tb}_{l}_align{a}".format(
          p = self.prefix,
          op = opcode_class_name,
          ex = self.extended_name(),
          tb = threadblock,
          l = self.layout_name(),
          a = str(self.A.alignment))

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.procedural_name()


###################################################################################################
#
# Data structure modeling a grouped GEMM operation
#
###################################################################################################

#
class GroupedGemmOperation(GemmOperation):
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, \
      scheduler_mode = GroupScheduleMode.Device):
    super().__init__(gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
                     epilogue_functor, swizzling_functor)

    self.scheduler_mode = scheduler_mode

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    base = super().procedural_name()
    return SubstituteTemplate(
      base + "_schedule${schedule}",
      {
        'schedule': ShortGroupScheduleModeNames[self.scheduler_mode]
      })


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""
    self.gemm_complex_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::GemmComplex<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${transform_a},
    ${transform_b},
    ${math_operation}
    ${residual}
  >;
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    residual = ''
    
    values = {
      'operation_name': operation.procedural_name(),
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
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'residual': residual
    }

    template = self.gemm_complex_template if operation.is_complex() else self.gemm_template

    return SubstituteTemplate(template, values)

###################################################################################################

class EmitSparseGemmInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::SparseGemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    residual = ''
    
    values = {
      'operation_name': operation.procedural_name(),
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
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'residual': residual
    }

    template = self.gemm_template

    return SubstituteTemplate(template, values)

###################################################################################################


#
class EmitGemmUniversalInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/arch/arch.h",
      "cutlass/arch/mma.h",
      "cutlass/layout/matrix.h",
      "cutlass/gemm/device/gemm.h",
      "cutlass/gemm/device/gemm_universal_adapter.h",
      "cutlass/gemm/kernel/default_gemm_universal.h",
    ]
    self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
    self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},    // transposed B operand
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},    // transposed A operand
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} : 
  public ${operation_name}_base { };
"""
    self.gemm_template_interleaved = """
// Gemm operator ${operation_name}
using ${operation_name}_base = 
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} : 
  public ${operation_name}_base { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
      cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
    >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    if operation.A.layout in transpose_layouts.keys() and \
      operation.B.layout in transpose_layouts.keys() and \
      operation.C.layout in transpose_layouts.keys():

      instance_layout_A = transpose_layouts[operation.A.layout]
      instance_layout_B = transpose_layouts[operation.B.layout]
      instance_layout_C = transpose_layouts[operation.C.layout]

      gemm_template = self.gemm_template
    else:
      instance_layout_A, instance_layout_B, instance_layout_C = \
        (operation.A.layout, operation.B.layout, operation.C.layout)

      gemm_template = self.gemm_template_interleaved
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #

    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
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
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(gemm_template, values)


###################################################################################################

#
class EmitGemmUniversal3xInstance:
  ''' Responsible for emitting a CUTLASS 3.x template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/gemm/gemm.h",
      "cutlass/numeric_types.h",
      "cutlass/gemm/kernel/gemm_universal.hpp",
      "cutlass/gemm/collective/collective_builder.hpp",
      "cutlass/epilogue/collective/collective_builder.hpp",
    ]
    self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
    self.gemm_template = """

using ${operation_name}_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule}
  >::CollectiveOp;

using ${operation_name}_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      sizeof(typename ${operation_name}_epilogue::SharedStorage)>,
  ${kernel_schedule}
  >::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    ${operation_name}_mainloop,
    ${operation_name}_epilogue>;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };

"""
  #
  def instance_template(self):
    return """
${compile_guard_start}
  using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
  manifest.append(
    new ${gemm_kind}<GemmKernel>("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    tile_shape = operation.tile_description.tile_shape
    warp_count = operation.tile_description.warp_count
    # stage count set to zero indicates builder automatic stage selection
    if operation.tile_description.stages > 0:
      stage_count_string = f"cutlass::gemm::collective::StageCount<{str(operation.tile_description.stages)}>"
    else:
      stage_count_string = "cutlass::gemm::collective::StageCountAuto"
    warp_shape = [tile_shape[idx] // warp_count[idx] for idx in range(3)]

    instance_layout_A, instance_layout_B, instance_layout_C , instance_layout_D = \
      (operation.A.layout, operation.B.layout, operation.C.layout, operation.D.layout)

    # 3.0 profiler integration only supports trivial epilogues for now
    epilogue_vector_length = 1

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):
      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #

    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_d': DataTypeTag[operation.D.element],
      'layout_d': LayoutTag[instance_layout_D],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': "cutlass::arch::Sm%d" % operation.arch,
      'tile_shape_m': str(operation.tile_description.tile_shape[0]),
      'tile_shape_n': str(operation.tile_description.tile_shape[1]),
      'tile_shape_k': str(operation.tile_description.tile_shape[2]),
      'cluster_m': str(operation.tile_description.cluster_shape[0]),
      'cluster_n': str(operation.tile_description.cluster_shape[1]),
      'cluster_k': str(operation.tile_description.cluster_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'kernel_schedule' : str(KernelScheduleTag[operation.kernel_schedule]),
      'epilogue_schedule' : str(EpilogueScheduleTag[operation.epilogue_schedule]),
      'epilogue_functor': epilogue_functor,
      'stages': stage_count_string,
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'align_d': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
    }

    return SubstituteTemplate(self.gemm_template, values)

###################################################################################################

#
class EmitGemmPlanarComplexInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${alignment_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${alignment_b},
    ${element_c}, cutlass::layout::RowMajor,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      ${element_c},
      ${alignment_c},
      ${element_accumulator},
      ${element_epilogue}
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator}
  >::GemmKernel;

  struct ${operation_name} : 
    public Operation_${operation_name} { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    # exchange and transpose A and B types, layouts, and complex transforms since the C layout is row-major
    transposed_layout_A = TransposedLayout[operation.A.layout]
    transposed_layout_B = TransposedLayout[operation.B.layout]

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.B.element],
      'layout_a': LayoutTag[transposed_layout_B],
      'transform_a': ComplexTransformTag[operation.B.complex_transform],
      'alignment_a': str(operation.B.alignment),
      'element_b': DataTypeTag[operation.A.element],
      'layout_b': LayoutTag[transposed_layout_A],
      'transform_b': ComplexTransformTag[operation.A.complex_transform],
      'alignment_b': str(operation.A.alignment),
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.tile_description.math_instruction.element_accumulator],
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
      'alignment_c': str(operation.C.alignment),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'stages': str(operation.tile_description.stages),
      'math_operator': 'cutlass::arch::OpMultiplyAdd'
    }

    return SubstituteTemplate(self.template, values)

###################################################################################################

#
class EmitGemmPlanarComplexArrayInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = typename cutlass::gemm::kernel::DefaultGemmPlanarComplexUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${alignment_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${alignment_b},
    ${element_c}, cutlass::layout::RowMajor,
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
      ${element_c},
      ${alignment_c},
      ${element_accumulator},
      ${element_epilogue}
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator}
  >::GemmArrayKernel;

  struct ${operation_name} : public Operation_${operation_name} { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    # exchange and transpose A and B types, layouts, and complex transforms since the C layout is row-major
    transposed_layout_A = TransposedLayout[operation.A.layout]
    transposed_layout_B = TransposedLayout[operation.B.layout]

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.B.element],
      'layout_a': LayoutTag[transposed_layout_B],
      'transform_a': ComplexTransformTag[operation.B.complex_transform],
      'alignment_a': str(operation.B.alignment),
      'element_b': DataTypeTag[operation.A.element],
      'layout_b': LayoutTag[transposed_layout_A],
      'transform_b': ComplexTransformTag[operation.A.complex_transform],
      'alignment_b': str(operation.A.alignment),
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.tile_description.math_instruction.element_accumulator],
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
      'alignment_c': str(operation.C.alignment),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'stages': str(operation.tile_description.stages),
      'math_operator': 'cutlass::arch::OpMultiplyAdd'
    }

    return SubstituteTemplate(self.template, values)

###################################################################################################

#
class EmitGemmGroupedInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/arch/arch.h",
      "cutlass/arch/mma.h",
      "cutlass/layout/matrix.h",
      "cutlass/gemm/device/gemm.h",
      "cutlass/gemm/kernel/gemm_grouped.h",
      "cutlass/gemm/kernel/default_gemm_grouped.h",
      "cutlass/gemm/device/gemm_grouped.h"
    ]
    self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
    self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${scheduler_mode},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
    cutlass::gemm::device::GemmGrouped<${operation_name}>
  >("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    instance_layout_A, instance_layout_B, instance_layout_C = \
      (operation.A.layout, operation.B.layout, operation.C.layout)
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #

    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
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
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'scheduler_mode': GroupScheduleModeTag[operation.scheduler_mode],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(self.gemm_template, values)

###################################################################################################
#
# Emitters functions for all targets
#
###################################################################################################

class EmitGemmConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name).replace('\\', '/')

    self.instance_emitter = {
      GemmKind.Gemm: EmitGemmInstance,
      GemmKind.Sparse: EmitSparseGemmInstance,
      GemmKind.Universal: EmitGemmUniversalInstance,
      GemmKind.Universal3x: EmitGemmUniversal3xInstance,
      GemmKind.PlanarComplex: EmitGemmPlanarComplexInstance,
      GemmKind.PlanarComplexArray: EmitGemmPlanarComplexArrayInstance,
      GemmKind.Grouped: EmitGemmGroupedInstance
    }

    self.gemm_kind_wrappers = {
      GemmKind.Gemm: 'GemmOperation',
      GemmKind.Sparse: 'GemmSparseOperation',
      GemmKind.Universal: 'GemmUniversalOperation',
      GemmKind.Universal3x: 'GemmUniversal3xOperation',
      GemmKind.PlanarComplex: 'GemmPlanarComplexOperation',
      GemmKind.PlanarComplexArray: 'GemmPlanarComplexArrayOperation',
      GemmKind.Grouped: 'GemmGroupedOperation'
    }

    self.wmma_guard_start = "#if defined(CUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)"

    self.separator = """
///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/
"""

    self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
    self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  def __enter__(self):
    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(self.header_template)
    self.configuration_file.write(self.separator)

    self.includes = collections.OrderedDict([
      ("cutlass/cutlass.h", None),
      ("cutlass/library/library.h", None),
      ("cutlass/library/manifest.h", None),
      ("library_internal.h", None),
      ("gemm_operation.h", None),
      ("gemm_operation_3x.hpp", None),
      ("cutlass/arch/wmma.h", None),
      ("cutlass/numeric_types.h", None)
    ])
    self.instance_definitions = []
    self.instance_wrappers = []

    self.operations = []
    return self

  def emit(self, operation):
    emitter = self.instance_emitter[operation.gemm_kind]()

    for incl in emitter.includes:
      self.includes[incl] = None

    self.operations.append(operation)

    self.instance_definitions.append(emitter.emit(operation))

    self.instance_wrappers.append(SubstituteTemplate(emitter.instance_template(), {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'gemm_kind': self.gemm_kind_wrappers[operation.gemm_kind],
      'compile_guard_start': SubstituteTemplate(self.wmma_guard_start, {'sm_number': str(operation.arch)}) \
        if operation.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp else "",
      'compile_guard_end': "#endif" \
        if operation.tile_description.math_instruction.opcode_class == OpcodeClass.WmmaTensorOp else "" 
      }))

  def __exit__(self, exception_type, exception_value, traceback):

    # Write includes
    for incl, _ in self.includes.items():
      include_statement = "#include \"%s\"\n" % incl
      self.configuration_file.write(include_statement)

    self.configuration_file.write(self.separator)

    # Write instance definitions in top-level namespace
    for instance_definition in self.instance_definitions:
      self.configuration_file.write(instance_definition)

    # Add wrapper objects within initialize() function
    self.configuration_file.write(SubstituteTemplate(self.initialize_function_template, {
      'configuration_name': self.configuration_name
      }))
   
    for instance_wrapper in self.instance_wrappers:
      self.configuration_file.write(instance_wrapper) 

    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()

###################################################################################################
###################################################################################################


class EmitGemmSoftmaxInstance:
  def __init__(self, operation_suffix = ''):
      self.operation_suffix = operation_suffix
      self.includes = [
        "cutlass/cutlass.h",
        "cutlass/numeric_types.h",
        "cutlass/arch/arch.h",
        "cutlass/arch/mma.h",
        "cutlass/layout/matrix.h",
        "cutlass/gemm/device/gemm.h",
        "cutlass/gemm/kernel/gemm_grouped.h",
        "cutlass/gemm/kernel/default_gemm_grouped.h",
        "cutlass/gemm/device/gemm_grouped.h",
        "cutlass/gemm/kernel/default_gemm.h"
      ]

      self.builtin_epilogue_functor_template = """
          ${epilogue_functor}<
            ${element_c},
            ${epilogue_vector_length},
            ${element_accumulator},
            ${element_epilogue}
          >
      """

      self.gemm_template = """
      
    using ${operation_name}_base =
    cutlass::GemmSoftmax<
        ${element_a}, ${layout_a}, ${align_a},
        ${element_b}, ${layout_b}, ${align_b},
        ${element_c}, ${align_c},
        ${opcode_class},
        ${arch},
        ${element_accumulator},
        ${stages},
        cutlass::gemm::GemmShape<${threadblock_shape_m},
                                  ${threadblock_shape_n},
                                  ${threadblock_shape_k}>,
        cutlass::gemm::GemmShape<${warp_shape_m},
                                  ${warp_shape_n},
                                  ${warp_shape_k}>,
        cutlass::gemm::GemmShape<${instruction_shape_m},
                                  ${instruction_shape_n},
                                  ${instruction_shape_k}>,
        ${epilogue_functor},
        ${swizzling_functor}
    >;
      """


  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    instance_layout_A, instance_layout_B, instance_layout_C =       (operation.A.layout, operation.B.layout, operation.C.layout)
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length =         min(operation.C.alignment * DataTypeSize[operation.C.element], 128)           // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()


    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
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
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(self.gemm_template, values)


class EmitGemmPermuteInstance:
  def __init__(self, operation_suffix = ''):
      self.operation_suffix = operation_suffix
      self.includes = []
      self.gemm_template = """
      
    using ${operation_name}_base =
    cutlass::gemm::device::GemmUniversal<
        ${element_a}, ${layout_a},
        ${element_b}, ${layout_b},
        ${element_c}, ${layout_c},
        ${element_accumulator},
        ${opcode_class},
        ${arch},
        cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
        cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
        cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
        ${epilogue_functor}<
          ${element_c},
          ${epilogue_vector_length},
          ${element_accumulator},
          ${element_epilogue}
        >,
        ${swizzling_functor},
        ${stages},
        ${align_a},
        ${align_b},
        ${math_operation},
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone,
        false,  /*GatherA*/
        false,  /*GatherB*/
        false,  /*ScatterD*/
        ${permute_layout} /*PermuteDLayout*/
    >;
      """


  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    values = {
      'operation_name': operation.procedural_name(),
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
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'permute_layout': EpiloguePermuteLayoutTag[operation.permute_layout],
    }

    return SubstituteTemplate(self.gemm_template, values)


class EmitDualGemmInstance:
  def __init__(self, operation_suffix = ''):
      self.operation_suffix = operation_suffix
      self.includes = []
      self.builtin_epilogue_functor_template = """
          ${epilogue_functor}<
            ${element_c},
            ${epilogue_vector_length},
            ${element_accumulator},
            ${element_epilogue},
            cutlass::epilogue::thread::ScaleType::Nothing
          >,
      """
      self.builtin_epilogue_functor2_template = """
          ${epilogue_functor2}<
            ${element_c},
            ${epilogue_vector_length},
            ${element_c},
            ${element_epilogue}
          >,
      """

      self.gemm_template = """
      
    using ${operation_name}_base =
    cutlass::gemm::device::DualGemm<
        ${element_a},
        ${layout_a},
        ${element_b},
        ${layout_b0},
        ${layout_b1},
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
                                  ${warp_shape_k}>,
        cutlass::gemm::GemmShape<${instruction_shape_m},
                                  ${instruction_shape_n},
                                  ${instruction_shape_k}>,
        ${epilogue_functor}
        ${epilogue_functor}
        ${epilogue_functor2}
        ${swizzling_functor},
        ${stages},
        false,
        false,
        false
    >;
      """


  def emit(self, operation, broadcast_b1=False):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    instance_layout_A, instance_layout_B0, instance_layout_C =       (operation.A.layout, operation.B.layout, operation.C.layout)
    # B1 is broadcasted in column-major with zero stride (the latter set in the Arguments)
    instance_layout_B1 = LayoutType.ColumnMajor if broadcast_b1 else instance_layout_B0

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length =         min(operation.C.alignment * DataTypeSize[operation.C.element], 128)           // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
        'epilogue_functor2': EpilogueFunctorTag[operation.epilogue_functor2],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
      epilogue_functor2 = SubstituteTemplate(self.builtin_epilogue_functor2_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
      epilogue_functor2 = self.epilogue_functor.emit_declaration()


    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b0': LayoutTag[instance_layout_B0],
      'layout_b1': LayoutTag[instance_layout_B1],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
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
      'epilogue_functor': epilogue_functor,
      'epilogue_functor2': epilogue_functor2,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(self.gemm_template, values)

