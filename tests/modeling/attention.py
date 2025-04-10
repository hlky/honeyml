import unittest
from typing import cast, List, Optional

import diffusers.models.attention_processor as attention_processor_torch
import torch

from honey.compiler import compile_model
from honey.frontend import nn, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import get_random_torch_tensor

import honey.modeling.diffusers.attention_processor as attention_processor
from honey.builder.config import mark_output


class AttentionTestCase(unittest.TestCase):
    def _test_attention(
        self,
        hidden_shape: List[int],
        query_dim: int,
        encoder_shape: Optional[List[int]] = None,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        out_dim: int = None,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        if len(hidden_shape) == 4:
            batch, channel, height, width = hidden_shape
        hidden_states = get_random_torch_tensor(hidden_shape, dtype=dtype)
        encoder_hidden_states = (
            get_random_torch_tensor(encoder_shape, dtype=dtype)
            if encoder_shape
            else None
        )
        attention_mask = (
            get_random_torch_tensor(
                [hidden_shape[0], hidden_shape[1]], dtype=dtype
            ).round()
            if cross_attention_norm
            else None
        )

        hidden_states_honey = hidden_states.clone().to(
            hidden_states.device, hidden_states.dtype
        )
        if len(hidden_shape) == 4:
            hidden_states_honey = hidden_states_honey.permute(0, 2, 3, 1).contiguous()
        encoder_hidden_states_honey = (
            encoder_hidden_states.clone().to(
                encoder_hidden_states.device, encoder_hidden_states.dtype
            )
            if encoder_hidden_states is not None
            else None
        )
        attention_mask_honey = (
            attention_mask.clone().to(attention_mask.device, attention_mask.dtype)
            if attention_mask is not None
            else None
        )

        op = (
            attention_processor_torch.Attention(
                query_dim=query_dim,
                cross_attention_dim=cross_attention_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                bias=bias,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                cross_attention_norm=cross_attention_norm,
                cross_attention_norm_num_groups=cross_attention_norm_num_groups,
                added_kv_proj_dim=added_kv_proj_dim,
                norm_num_groups=norm_num_groups,
                spatial_norm_dim=spatial_norm_dim,
                out_bias=out_bias,
                scale_qk=scale_qk,
                only_cross_attention=only_cross_attention,
                eps=eps,
                rescale_output_factor=rescale_output_factor,
                residual_connection=residual_connection,
                out_dim=out_dim,
            )
            .eval()
            .to(hidden_states.device, hidden_states.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(hidden_states.device, hidden_states.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt = op.forward(hidden_states, encoder_hidden_states, attention_mask)

        y = torch.empty_like(y_pt).to(hidden_states.device, hidden_states.dtype)
        if len(y.shape) == 4:
            y = y.permute(0, 2, 3, 1).contiguous()

        Hidden_states = Tensor(
            shape=(
                [batch, height, width, channel]
                if len(hidden_shape) == 4
                else hidden_shape
            ),
            dtype=dtype,
            name="Hidden_states",
            is_input=True,
        )
        Encoder_hidden_states = (
            Tensor(
                shape=encoder_shape,
                dtype=dtype,
                name="Encoder_hidden_states",
                is_input=True,
            )
            if encoder_hidden_states is not None
            else None
        )
        Attention_mask = (
            Tensor(
                shape=[hidden_shape[0], hidden_shape[1]],
                dtype=dtype,
                name="Attention_mask",
                is_input=True,
            )
            if attention_mask is not None
            else None
        )

        op_honey = attention_processor.Attention(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cross_attention_norm=cross_attention_norm,
            cross_attention_norm_num_groups=cross_attention_norm_num_groups,
            added_kv_proj_dim=added_kv_proj_dim,
            norm_num_groups=norm_num_groups,
            spatial_norm_dim=spatial_norm_dim,
            out_bias=out_bias,
            scale_qk=scale_qk,
            only_cross_attention=only_cross_attention,
            eps=eps,
            rescale_output_factor=rescale_output_factor,
            residual_connection=residual_connection,
            out_dim=out_dim,
            dtype=dtype,
        )
        op_honey.name_parameter_tensor()
        if Encoder_hidden_states is not None and Attention_mask is not None:
            Y = op_honey.forward(Hidden_states, Encoder_hidden_states, Attention_mask)
        elif Encoder_hidden_states is not None:
            Y = op_honey.forward(Hidden_states, Encoder_hidden_states)
        elif Attention_mask is not None:
            Y = op_honey.forward(Hidden_states, attention_mask=Attention_mask)
        else:
            Y = op_honey.forward(Hidden_states)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_attention_{dtype}_dim{query_dim}_heads{heads}_dim_head{dim_head}"
        )
        inputs_dict = {"Hidden_states": hidden_states_honey}
        if Encoder_hidden_states is not None:
            inputs_dict["Encoder_hidden_states"] = encoder_hidden_states_honey
        if Attention_mask is not None:
            inputs_dict["Attention_mask"] = attention_mask_honey
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs_dict, [y])
        if y.ndim == 4:
            y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def test_attention(self):
        self._test_attention(
            hidden_shape=[1, 5632, 320],
            encoder_shape=[1, 77, 768],
            cross_attention_dim=768,
            query_dim=320,
            heads=8,
            dim_head=40,
            bias=True,
            dtype="float16",
            tolerance=1e-3,
        )
        self._test_attention(
            hidden_shape=[1, 512, 64, 88],
            query_dim=512,
            heads=1,
            dim_head=512,
            bias=True,
            dtype="float16",
            tolerance=1e-3,
        )
        self._test_attention(
            hidden_shape=[1, 88, 1280],
            query_dim=1280,
            heads=8,
            dim_head=160,
            bias=True,
            dtype="float16",
            tolerance=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
