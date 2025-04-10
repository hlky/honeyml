import unittest

from typing import cast, List, Optional, Tuple

import diffusers.models.normalization as normalization_torch

import torch
from honey.compiler import compile_model, ops
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.testing.test_utils import get_random_torch_tensor

import honey.modeling.diffusers.normalization as normalization
from honey.builder.config import mark_output



class NormalizationTestCase(unittest.TestCase):
    def _test_rms_norm(
        self,
        shape: List[int],
        hidden_size: int,
        eps: float,
        elementwise_affine: bool,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_honey = x.clone().to(x.device, x.dtype)

        op = (
            normalization_torch.RMSNorm(
                dim=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt: torch.Tensor = op.forward(x)
        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op = normalization.RMSNorm(
            dim=hidden_size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_rms_norm_{dtype}_dim{hidden_size}_eps{eps}"
        if elementwise_affine:
            test_name += "_elementwise_affine"
        inputs = {"X": x_honey}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def _test_ada_layer_norm(
        self,
        shape: List[int],
        embedding_dim: int,
        num_embeddings: int,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        # NOTE: Diffusers' AdaLayerNorm expects rank 0 for timestep
        timestep = torch.randint(69, 420, [], device=x.device).to(torch.int64)
        x_honey = x.clone()
        timestep_honey = timestep.clone().unsqueeze(0)

        op = (
            normalization_torch.AdaLayerNorm(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt: torch.Tensor = op.forward(x, timestep)

        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Timestep = Tensor(
            shape=[1],
            dtype="int64",
            name="timestep",
            is_input=True,
        )

        op = normalization.AdaLayerNorm(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X, Timestep)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_ada_layer_norm_{dtype}_dim{embedding_dim}_number{num_embeddings}"
        )
        inputs = {"X": x_honey, "timestep": timestep_honey}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def _test_ada_layer_norm_zero(
        self,
        shape: List[int],
        hidden_size: int,
        num_embeddings: int,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, _, _ = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_honey = x.clone()
        timestep = get_random_torch_tensor([batch], dtype=dtype)
        timestep_honey = timestep.clone()
        class_labels = torch.randint(
            69, 420, [num_embeddings], device=timestep.device
        ).to(torch.int64)
        class_labels_honey = class_labels.clone().to(
            class_labels.device, class_labels.dtype
        )

        op = (
            normalization_torch.AdaLayerNormZero(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_size,
            )
            .eval()
            .to(timestep.device, timestep.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(timestep.device, timestep.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            outputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = op.forward(
                x, timestep, class_labels, hidden_dtype=timestep.dtype
            )
        outputs_honey = {
            "Y": torch.empty_like(outputs[0]),
            "gate_msa": torch.empty_like(outputs[1]),
            "shift_mlp": torch.empty_like(outputs[2]),
            "scale_mlp": torch.empty_like(outputs[3]),
            "gate_mlp": torch.empty_like(outputs[4]),
        }
        for key, tensor in outputs_honey.items():
            print(f"{key} - {tensor.shape}")

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Timestep = Tensor(
            shape=[batch],
            dtype=dtype,
            name="timestep",
            is_input=True,
        )
        ClassLabels = Tensor(
            shape=[num_embeddings],
            dtype="int64",
            name="class_labels",
            is_input=True,
        )

        op = normalization.AdaLayerNormZero(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_size,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Outputs = op.forward(X, Timestep, ClassLabels)
        Outputs = [
            mark_output(Outputs[0], "Y"),
            mark_output(Outputs[1], "gate_msa"),
            mark_output(Outputs[2], "shift_mlp"),
            mark_output(Outputs[3], "scale_mlp"),
            mark_output(Outputs[4], "gate_mlp"),
        ]

        target = detect_target()
        test_name = (
            f"test_ada_layer_norm_zero_{dtype}_num{num_embeddings}_dim{hidden_size}"
        )
        inputs = {
            "X": x_honey,
            "timestep": timestep_honey,
            "class_labels": class_labels_honey,
        }
        module = compile_model(
            Outputs,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, outputs_honey)
        for idx, name in enumerate(outputs_honey.keys()):
            y: torch.Tensor = outputs_honey[name]
            y_pt: torch.Tensor = outputs[idx]
            torch.testing.assert_close(
                y,
                y_pt.to(y.dtype),
                rtol=tolerance,
                atol=tolerance,
                msg=lambda msg: f"{msg}\n\n{name}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
            )

    def _test_ada_layer_norm_single(
        self,
        hidden_size: int,
        use_additional_conditions: bool,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        timestep = get_random_torch_tensor([1], dtype=dtype)
        timestep_honey = timestep.clone()

        op = (
            normalization_torch.AdaLayerNormSingle(
                embedding_dim=hidden_size,
                use_additional_conditions=use_additional_conditions,
            )
            .eval()
            .to(timestep.device, timestep.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(timestep.device, timestep.dtype)
            state_dict_honey[key_honey] = value

        batch_size = timestep.shape[0]
        height, width = 512, 512
        if use_additional_conditions:
            resolution = (
                torch.tensor([height, width])
                .repeat(batch_size, 1)
                .to(timestep.device, timestep.dtype)
            )
            aspect_ratio = (
                torch.tensor([float(height / width)])
                .repeat(batch_size, 1)
                .to(timestep.device, timestep.dtype)
            )
        else:
            resolution = None
            aspect_ratio = None

        kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
        with torch.inference_mode():
            outputs: Tuple[torch.Tensor, torch.Tensor] = op.forward(
                timestep, kwargs, batch_size, hidden_dtype=timestep.dtype
            )
        outputs_honey = {
            "Y": torch.empty_like(outputs[0]),
            "embedded_timestep": torch.empty_like(outputs[1]),
        }
        for key, tensor in outputs_honey.items():
            print(f"{key} - {tensor.shape}")

        X = Tensor(shape=[1], dtype=dtype, name="X", is_input=True)
        Resolution = Tensor(
            shape=[batch_size, 2], dtype=dtype, name="resolution", is_input=True
        )
        AspectRatio = Tensor(
            shape=[batch_size, 1], dtype=dtype, name="aspect_ratio", is_input=True
        )

        op = normalization.AdaLayerNormSingle(
            embedding_dim=hidden_size,
            use_additional_conditions=use_additional_conditions,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Outputs = op.forward(X, Resolution, AspectRatio)
        Outputs = [
            mark_output(Outputs[0], "Y"),
            mark_output(Outputs[1], "embedded_timestep"),
        ]

        target = detect_target()
        test_name = f"test_ada_layer_norm_single_{dtype}_dim{hidden_size}_added-cond{use_additional_conditions}"
        inputs = {"X": timestep_honey}
        if use_additional_conditions:
            inputs.update({"resolution": resolution, "aspect_ratio": aspect_ratio})
        module = compile_model(
            Outputs,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, outputs_honey)
        for idx, name in enumerate(outputs_honey.keys()):
            y: torch.Tensor = outputs_honey[name]
            y_pt: torch.Tensor = outputs[idx]
            torch.testing.assert_close(
                y,
                y_pt.to(y.dtype),
                rtol=tolerance,
                atol=tolerance,
                msg=lambda msg: f"{msg}\n\n{name}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
            )

    def _test_ada_group_norm(
        self,
        shape: List[int],
        embedding_dim: int,
        out_dim: int,
        num_groups: int,
        act_fn: Optional[str] = None,
        eps: float = 1e-5,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        b, c, h, w = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        emb = get_random_torch_tensor([shape[0], embedding_dim], dtype=dtype)
        x_honey = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)
        emb_honey = emb.clone().to(emb.device, emb.dtype)

        op = (
            normalization_torch.AdaGroupNorm(
                embedding_dim=embedding_dim,
                out_dim=out_dim,
                num_groups=num_groups,
                act_fn=act_fn,
                eps=eps,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            value = value.to(x.device, x.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt = op.forward(x, emb)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[b, h, w, c],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Emb = Tensor(
            shape=[shape[0], embedding_dim],
            dtype=dtype,
            name="Emb",
            is_input=True,
        )

        op_honey = normalization.AdaGroupNorm(
            embedding_dim=embedding_dim,
            out_dim=out_dim,
            num_groups=num_groups,
            act_fn=act_fn,
            eps=eps,
        )
        op_honey.name_parameter_tensor()
        Y = op_honey.forward(X, Emb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_ada_group_norm_{dtype}_dim{out_dim}_groups{num_groups}_eps{eps}"
        )
        if act_fn:
            test_name += f"_{act_fn}"
        inputs = {"X": x_honey, "Emb": emb_honey}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def _test_ada_layer_norm_continuous(
        self,
        shape: List[int],
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        conditioning_embedding = get_random_torch_tensor(
            [shape[0], conditioning_embedding_dim], dtype=dtype
        )
        x_honey = x.clone().to(x.device, x.dtype)
        conditioning_embedding_honey = conditioning_embedding.clone().to(
            conditioning_embedding.device, conditioning_embedding.dtype
        )

        op = (
            normalization_torch.AdaLayerNormContinuous(
                embedding_dim=embedding_dim,
                conditioning_embedding_dim=conditioning_embedding_dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
                bias=bias,
                norm_type=norm_type,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            value = value.to(x.device, x.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt = op.forward(x, conditioning_embedding)

        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        ConditioningEmbedding = Tensor(
            shape=[shape[0], conditioning_embedding_dim],
            dtype=dtype,
            name="ConditioningEmbedding",
            is_input=True,
        )

        op_honey = normalization.AdaLayerNormContinuous(
            embedding_dim=embedding_dim,
            conditioning_embedding_dim=conditioning_embedding_dim,
            elementwise_affine=elementwise_affine,
            eps=eps,
            bias=bias,
            norm_type=norm_type,
            dtype=dtype,
        )
        op_honey.name_parameter_tensor()
        Y = op_honey.forward(X, ConditioningEmbedding)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_ada_layer_norm_continuous_{dtype}_dim{embedding_dim}_conditioning_dim{conditioning_embedding_dim}_norm{norm_type}_eps{eps}"
        inputs = {"X": x_honey, "ConditioningEmbedding": conditioning_embedding_honey}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def _test_global_response_norm(
        self,
        shape: List[int],
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_honey = x.clone().to(x.device, x.dtype)

        op = (
            normalization_torch.GlobalResponseNorm(dim=shape[-1])
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            value = value.to(x.device, x.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt: torch.Tensor = op.forward(x)

        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op_honey = normalization.GlobalResponseNorm(dim=shape[-1], dtype=dtype)
        op_honey.name_parameter_tensor()
        Y = op_honey.forward(X)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_global_response_norm_{dtype}_dim{shape[-1]}"
        inputs = {"X": x_honey}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_pt,
        )
        module.run_with_tensors(inputs, [y])
        y = y
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def test_rms_norm(self):
        self._test_rms_norm(
            shape=[1, 13, 768],
            hidden_size=768,
            eps=1e-6,
            elementwise_affine=True,
            tolerance=1e-3,
            dtype="float16",
        )
        self._test_rms_norm(
            shape=[1, 13, 768],
            hidden_size=768,
            eps=1e-6,
            elementwise_affine=False,
            tolerance=1e-3,
            dtype="float16",
        )

    def test_ada_layer_norm(self):
        self._test_ada_layer_norm(
            shape=[1, 13, 1152],
            embedding_dim=1152,
            num_embeddings=1000,
            tolerance=4e-3,
            dtype="float16",
        )
        self._test_ada_layer_norm(
            shape=[2, 13, 1152],
            embedding_dim=1152,
            num_embeddings=1000,
            tolerance=4e-3,
            dtype="float16",
        )

    def test_ada_layer_norm_zero(self):
        self._test_ada_layer_norm_zero(
            shape=[1, 1000, 1152],
            hidden_size=1152,
            num_embeddings=1000,
            tolerance=4e-3,
            dtype="float16",
        )

    def test_ada_layer_norm_single(self):
        self._test_ada_layer_norm_single(
            hidden_size=1152,
            use_additional_conditions=False,
            tolerance=1e-3,
            dtype="float16",
        )

    def test_ada_group_norm(self):
        self._test_ada_group_norm(
            shape=[1, 384, 64, 64],
            embedding_dim=768,
            out_dim=384,
            num_groups=768 // 32,
            tolerance=2e-3,
            dtype="float16",
        )

    def test_ada_layer_norm_continuous(self):
        self._test_ada_layer_norm_continuous(
            shape=[1, 13, 768],
            embedding_dim=768,
            conditioning_embedding_dim=128,
            elementwise_affine=True,
            eps=1e-5,
            bias=True,
            norm_type="layer_norm",
            tolerance=2e-3,
            dtype="float16",
        )

    def test_global_response_norm(self):
        self._test_global_response_norm(
            shape=[1, 64, 64, 32],
            tolerance=1e-3,
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
