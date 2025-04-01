import torch
from dataclasses import dataclass
from typing import (
    Tuple,
    Optional,
    Dict,
    Any,
    Annotated,
    get_type_hints,
    Union,
    Callable,
    List,
    get_origin,
    get_args,
)
import inspect

from ..frontend import IntImm, IntVar, Tensor


def get_device_name():
    cleanup = {
        "nvidia ": "",
        "geforce rtx ": "",
        "geforce gtx ": "",
        "geforce gt ": "",
        "geforce ": "",
        "tesla ": "",
        "quadro ": "",
        " ": "_",
    }
    split_by = {
        ",": 0,
        "(": 0,
    }

    device_name = torch.cuda.get_device_name().lower()
    for target, replacement in cleanup.items():
        device_name = device_name.replace(target, replacement).strip()

    for target, index in split_by.items():
        device_name = device_name.split(target)[index].strip()

    return device_name


def get_sm():
    sm = "".join(str(i) for i in torch.cuda.get_device_capability())
    return sm


@dataclass(frozen=True)
class DimOperation:
    value: Union[int, Callable[[Dict[str, int]], int]]

    def apply(self, x: int, config: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class DimAdd(DimOperation):
    def apply(self, x: int, config: Dict[str, int]) -> int:
        add_value = self.value(config) if callable(self.value) else self.value
        return x + add_value


@dataclass(frozen=True)
class DimSub(DimOperation):
    def apply(self, x: int, config: Dict[str, int]) -> int:
        sub_value = self.value(config) if callable(self.value) else self.value
        return x - sub_value


@dataclass(frozen=True)
class DimMul(DimOperation):
    def apply(self, x: int, config: Dict[str, int]) -> int:
        mul_value = self.value(config) if callable(self.value) else self.value
        return x * mul_value


@dataclass(frozen=True)
class DimDiv(DimOperation):
    def apply(self, x: int, config: Dict[str, int]) -> int:
        div_value = self.value(config) if callable(self.value) else self.value
        return x // div_value


@dataclass(frozen=True)
class Shape:
    name: str
    dim_operations: Tuple[DimOperation, ...] = ()
    config_name: Optional[str] = None

    def evaluate(
        self,
        symbolic_values: Dict[str, Union[int, Tuple[int, int]]],
        config: Dict[str, int],
    ) -> Optional[Union[IntImm, IntVar]]:
        if self.config_name:
            if self.config_name not in config:
                return None
            base = config[self.config_name]
        else:
            if self.name not in symbolic_values:
                raise ValueError(f"Missing value for shape '{self.name}'")
            base = symbolic_values[self.name]
        if isinstance(base, tuple):
            evaluated = tuple(self._apply_ops(b, config) for b in base)
            if evaluated[0] == evaluated[-1]:
                evaluated = evaluated[0]
        else:
            evaluated = self._apply_ops(base, config)
        if isinstance(evaluated, tuple):
            return IntVar(evaluated)
        else:
            return IntImm(evaluated)

    def _apply_ops(self, base: int, config: Dict[str, int]) -> int:
        for op in self.dim_operations:
            base = op.apply(base, config)
        return base


def build_tensors_from_annotations(
    forward_fn,
    symbolic_values: Dict[str, Union[int, List[int]]],
    config: Dict[str, int],
) -> Dict[str, Any]:
    tensors = {}
    sig = inspect.signature(forward_fn)
    type_hints = get_type_hints(forward_fn, include_extras=True)

    for param_name, param in sig.parameters.items():
        annotation = type_hints.get(param_name)
        if annotation is None:
            continue

        if get_origin(annotation) is Annotated:
            base_type, *metadata = get_args(annotation)
            if metadata and isinstance(metadata[0], dict):
                nested = metadata[0]
                nested_tensors = {}
                for key, shape_spec in nested.items():
                    if not isinstance(shape_spec, (list, tuple)):
                        shape_list = [shape_spec]
                    else:
                        shape_list = shape_spec
                    dims = []
                    remove_tensor = False
                    for shape in shape_list:
                        evaluated = shape.evaluate(symbolic_values, config)
                        if evaluated is not None:
                            dims.append(evaluated)
                        else:
                            remove_tensor = True
                    if dims and not remove_tensor:
                        nested_tensors[key] = Tensor(tuple(dims), name=key, is_input=True)
                if nested_tensors:
                    tensors[param_name] = nested_tensors
            elif metadata:
                shape_spec = metadata[0]
                if not isinstance(shape_spec, (list, tuple)):
                    shape_list = [shape_spec]
                else:
                    shape_list = shape_spec
                dims = []
                remove_tensor = False
                for shape in shape_list:
                    evaluated = shape.evaluate(symbolic_values, config)
                    if evaluated is not None:
                        dims.append(evaluated)
                    else:
                        remove_tensor = True
                if dims and not remove_tensor:
                    tensors[param_name] = Tensor(tuple(dims), name=param_name, is_input=True)

    return tensors
