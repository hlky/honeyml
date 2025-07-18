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
build a test module from a tensor
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from honey import backend, compiler
from honey.compiler.base import (
    DynamicProfileStrategy,
    IntImm,
    JaggedIntVar,
    Tensor,
)

from honey.compiler.model import (
    Honey_DEFAULT_NUM_RUNTIMES,
    HoneyAllocatorKind,
    Model,
    TorchTensor,
)
from honey.compiler.transform.name_graph import reset_name_counters
from honey.compiler.transform.profile import elapsed_dt_sec
from honey.utils import graph_utils
from honey.utils.debug_settings import HoneyDebugSettings
from honey.utils.misc import callstack_stats
from honey.utils.serialization.serdes_code import dump_program

# pylint: disable=W0102


_LOGGER = logging.getLogger(__name__)


def _validate_tensor_args(sorted_graph: List[Tensor], output_tensors: List[Tensor]):
    """
    Validate the user's desired output name -> index ordering.

    Errors if:
    1) The given ordering has duplicates
    2) The given ordering has non-outputs
    3) The given ordering is missing outputs that are reachable

    Note that we have to do this before any optimizations. It is legal to replace output tensors
    with new Tensor objects of the same name, so the user-provided tensors might not be in
    the graph after optimizations (replacing a Tensor sets is_output=False).
    """
    seen_tensors = set()
    for tensor in output_tensors:
        name = tensor._attrs["name"]
        if not tensor._attrs["is_output"]:
            raise ValueError(f"Got non-output tensor in output_tensors list: {name}")
        if name in seen_tensors:
            raise ValueError(f"Got duplicate name {name} in output_tensors list.")
        seen_tensors.add(name)

    given_tensors = {tensor._attrs["name"] for tensor in output_tensors}
    for tensor in reversed(sorted_graph):
        name = tensor._attrs["name"]
        if tensor._attrs["is_output"] and name not in given_tensors:
            raise ValueError(f"Output {name} was not passed into output_tensors")


def _verify_outputs_still_in_graph(sorted_graph: List[Tensor], outputs: List[Tensor]):
    seen = {tensor._attrs["name"]: False for tensor in outputs}
    for tensor in sorted_graph:
        name = tensor._attrs["name"]
        if name not in seen:
            continue

        if seen[name]:
            raise ValueError(
                f"Output {name} appears in the graph twice after optimizations."
            )

        seen[name] = True

    for tensor, was_seen in seen.items():
        if not was_seen:
            raise ValueError(
                f"Output {tensor} was not found in the graph after optimizations."
            )


def _mark_isolated_int_vars(sorted_graph: List[Tensor]):
    """
    Mark the IntVars that are not present in any input's shape
    with the _attrs["isolated"] = True flag. The purpose is to
    be able to distinguish these dynamic dims in the codegen
    of some of the functions which should set them instead of
    relying on / validating the pre-set value. To this end,
    this function must be invoked right before the back-end
    code generation of the ops.

    One example is the padded_dense_to_jagged op that must set
    the total_length dimension of the resulting jagged Tensor
    if it hasn't been set from any of the model input's shape.
    Another example is the make_jagged op that should set the
    batch_dim within the JaggedIntVar of the resulting jagged
    Tensor, unless it has been set already from the inputs.
    """
    int_vars = {}
    int_var_names_in_input_shapes = set()
    for tensor in sorted_graph:
        for dim in tensor._attrs["shape"]:
            if not isinstance(dim, IntImm):
                name = dim._attrs["name"]
                int_vars[name] = dim
                if isinstance(dim, JaggedIntVar):
                    batch_dim = dim.batch_dim()
                    if not isinstance(batch_dim, IntImm):
                        int_vars[batch_dim._attrs["name"]] = batch_dim
                    total_length = dim.total_length()
                    int_vars[total_length._attrs["name"]] = total_length
                    for jagged_dim in dim.jagged_dims():
                        min_value = jagged_dim.min_value()
                        if not isinstance(min_value, IntImm):
                            int_vars[min_value._attrs["name"]] = min_value
                        max_value = jagged_dim.max_value()
                        if not isinstance(max_value, IntImm):
                            int_vars[max_value._attrs["name"]] = max_value
                if tensor._attrs["is_input"]:
                    int_var_names_in_input_shapes.add(name)

    for name, dim in int_vars.items():
        if name not in int_var_names_in_input_shapes:
            dim._attrs["isolated"] = True


_DEBUG_SETTINGS = HoneyDebugSettings()


@callstack_stats()
def compile_model(
    tensor: Union[Tensor, List[Tensor]],
    target: backend.target.Target,
    workdir: str,
    test_name: str,
    profile_devs: List[int] = None,
    dynamic_profiling_strategy: DynamicProfileStrategy = DynamicProfileStrategy.MAX,
    dll_name: str = "test.so",
    num_runtimes: int = Honey_DEFAULT_NUM_RUNTIMES,
    profile_dir: str = None,
    constants: Optional[Dict[str, TorchTensor]] = None,
    allocator_kind: Optional[HoneyAllocatorKind] = None,
    debug_settings: HoneyDebugSettings = _DEBUG_SETTINGS,
    do_optimize_graph: bool = True,
    do_constant_folding: bool = True,
    profile_timeout: int = 3600,
    n_cpus: int = -1,
    do_compile: bool = True,
    do_build: bool = True,
) -> Model:
    """Compiles a model and generates a .so file.

    Parameters
    ----------
    tensor : Union[Tensor, List[Tensor]]
        An output Tensor, or a list of output Tensors.
        The compiled module will preserve the ordering of the outputs in its
        internal ordering.
    target : Target
        A compilation target. See comments for Target.
    workdir : str
        A workdir to store profiling and execution source codes, as well as the result .so file.
    test_name : str
        Name of the test. Used as the name of the subdir which stores the generated .so file.
    profile_devs : List[int], optional
        A list of profiling devices, by default device 0 will be used.
    dynamic_profiling_strategy: DynamicProfileStrategy, optional
        A DynamicProfileStrategy used for profiling. See comments for DynamicProfileStrategy.
    dll_name: str
        The output .so name.
    num_runtimes: int
        How many runtimes should be stored in the internal pool. This
        determines how many inferences can happen concurrently. By
        default, set to 1. Must be positive.
    profile_dir: str
        The base dir to generate profiling source codes. By default, workdir/test_name
    constants: Dict[str, TorchTensor], optional
        User-provided constants to bind to the graph. The constants can be folded and packaged into
        the final *.so.
    allocator_kind: HoneyAllocatorKind, optional
        The GPU allocator to use. If none is specified, use the default allocator.
    debug_settings: HoneyDebugSettings
        specify debug settings such as where to dump Honey model Python file, etc.
    do_optimize_graph: bool
        Apply full list of graph optimizations. Default: True

    Returns
    -------
    Model
        A model object.
    """
    if constants is None:
        constants = {}

    recompile = os.getenv("Honey_RECOMPILE", "1")
    graph = None
    os.makedirs(workdir, exist_ok=True)  # explicitly ensure workdir exists
    # Super important: we cannot have commas in the test name.
    # We want to add a -Iworkdir/test_name flag to nvcc, but
    # if the name has a comma in it, it will be parsed as two
    # arguments (even if we put quotes around it)!!
    test_name = test_name.replace(",", "_")
    test_dir = os.path.join(workdir, test_name)
    _LOGGER.info(f"Start to compile Honey model. {test_dir=}")
    if profile_dir is None:
        profile_dir = workdir

    if int(recompile) == 1:
        os.makedirs(test_dir, exist_ok=True)
        with target:
            reset_name_counters()
            graph = compiler.transform.toposort(tensor)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "toposort")

            output_tensors = [tensor] if isinstance(tensor, Tensor) else tensor
            _validate_tensor_args(graph, output_tensors)

            compiler.transform.bind_constants(graph, constants)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "bind_constants")

            compiler.transform.remove_unused_ops(graph)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "remove_unused_ops"
            )

            compiler.transform.remove_no_ops(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "remove_no_ops")

            compiler.transform.name_graph(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "name_graph")

            if debug_settings.dump_honey_to_py:
                dump_program(tensor, debug_settings.dump_honey_to_py)

            compiler.transform.dedup_symbolic_name(graph)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "dedup_symbolic_name"
            )

            compiler.transform.mark_param_tensor(graph)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "mark_param_tensor"
            )

            start_t = datetime.now()
            graph = compiler.transform.optimize_graph(
                graph, test_dir, optimize=do_optimize_graph
            )
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "optimize_graph")
            _LOGGER.info(f"optimized graph elapsed time: {elapsed_dt_sec(start_t)}")

            compiler.transform.mark_special_views(graph)
            compiler.transform.refine_graph(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "refine_graph")

            if profile_devs is None:
                device_env = os.getenv(target.dev_select_flag(), None)
                if device_env is None:
                    profile_devs = [0]
                else:
                    profile_devs = device_env.split(",")
            compiler.transform.profile(
                graph,
                profile_dir,
                profile_devs,
                dynamic_profiling_strategy,
                profile_timeout,
            )
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "profile")

            start_t = datetime.now()
            constant_folding_workdir = os.path.join(workdir, test_name)
            os.makedirs(constant_folding_workdir, exist_ok=True)
            # TODO: investigate constant folding memory planning with some models
            (
                graph,
                constant_folding_file_pairs,
                constant_folding_inputs,
            ) = compiler.transform.constant_folding(
                graph, workdir, test_name, do_constant_folding=do_constant_folding
            )
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "constant_folding"
            )
            _LOGGER.info(f"folded constants elapsed time: {elapsed_dt_sec(start_t)}")

            compiler.transform.dedup_symbolic_name(graph)
            graph_utils.dump_graph_debug_str_to_file(
                graph, test_dir, "dedup_symbolic_name"
            )

            (
                max_blob,
                max_constant_blob,
                workspace,
            ) = compiler.transform.memory_planning(graph)
            _verify_outputs_still_in_graph(graph, output_tensors)
            _mark_isolated_int_vars(graph)
            graph_utils.dump_graph_debug_str_to_file(graph, test_dir, "memory_planning")

            file_pairs = backend.codegen.gen_function_src(graph, workdir, test_name)
            file_pairs.extend(constant_folding_file_pairs)

            # It's possible that the original output tensor has been replaced with a new tensor.
            # Preserve original output tensors' orders but use the new tensors.
            new_output_tensor_dict = {
                tensor._attrs["name"]: tensor
                for tensor in graph
                if tensor._attrs["is_output"]
            }
            output_tensors = [tensor] if isinstance(tensor, Tensor) else tensor
            output_tensors = [
                new_output_tensor_dict[tensor._attrs["name"]]
                for tensor in output_tensors
            ]

            main_pairs = backend.codegen.gen_library_src(
                graph,
                max_blob,
                max_constant_blob,
                workspace,
                workdir,
                output_tensors,
                test_name,
                additional_unbound_constants=constant_folding_inputs,
                debug_settings=debug_settings,
            )
            file_pairs.extend(main_pairs)

            if do_compile:
                start_t = datetime.now()
                compile_engine = backend.builder.get_compile_engine(n_cpus)
                compile_engine.make(
                    file_pairs,
                    dll_name,
                    workdir,
                    test_name,
                    debug_settings,
                    do_build=do_build,
                )
                _LOGGER.info(
                    f"compiled the final .so file elapsed time: {elapsed_dt_sec(start_t)}",
                )
                if not do_build:
                    return
            else:
                return
    total_usage = max_blob + max_constant_blob + workspace.total_size()
    module = Model(
        os.path.join(workdir, test_name, dll_name), num_runtimes, allocator_kind
    )
    module.debug_sorted_graph = graph
    module.total_usage = total_usage
    return module
