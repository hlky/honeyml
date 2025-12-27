import math
from typing import Annotated, Any, Dict, List, Optional, Union

from dinoml.compiler import ops

from dinoml.frontend import IntVar, nn, Tensor
from dinoml.utils.shape_utils import get_shape

from ....utils.build_utils import Shape, DimAdd, DimDiv, DimMul, DimSub

from .configuration_t5 import T5Config
from ..modeling_outputs import BaseModelOutput
from ..activations import get_activation


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype: str = "float16"):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter([hidden_size], dtype=dtype)
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor):
        return ops.t5_layer_norm(
            hidden_states, self.weight.tensor(), eps=self.variance_epsilon
        )()


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config, dtype: str = "float16"):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False, dtype=dtype)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout_rate, dtype=dtype)
        if config.dense_act_fn == "gelu_new":
            self.act = ops.gelu_new()
        else:
            self.act = get_activation(config.dense_act_fn)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config, dtype: str = "float16"):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False, dtype=dtype)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False, dtype=dtype)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout_rate, dtype=dtype)
        if config.dense_act_fn == "gelu_new":
            self.act = ops.gelu_new()
        else:
            self.act = get_activation(config.dense_act_fn)

    def forward(self, hidden_states: Tensor):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config, dtype: str = "float16"):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config, dtype=dtype)
        else:
            self.DenseReluDense = T5DenseActDense(config, dtype=dtype)

        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, dtype=dtype
        )
        self.dropout = nn.Dropout(config.dropout_rate, dtype=dtype)

    def forward(self, hidden_states: Tensor):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False, dtype=dtype)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False, dtype=dtype)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False, dtype=dtype)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False, dtype=dtype)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                [self.relative_attention_num_buckets, self.n_heads], dtype=dtype
            )

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        query_length=None,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states._attrs["shape"][:2]

        query_states = self.q(hidden_states)
        query_states = ops.transpose()(
            ops.reshape()(
                query_states, [batch_size, -1, self.n_heads, self.key_value_proj_dim]
            ),
            1,
            2,
        )

        current_states = hidden_states

        key_states = self.k(current_states)
        value_states = self.v(current_states)
        key_states = ops.transpose()(
            ops.reshape()(
                key_states, [batch_size, -1, self.n_heads, self.key_value_proj_dim]
            ),
            1,
            2,
        )
        value_states = ops.transpose()(
            ops.reshape()(
                value_states, [batch_size, -1, self.n_heads, self.key_value_proj_dim]
            ),
            1,
            2,
        )
        BH = batch_size * self.n_heads
        S = seq_length
        D = self.key_value_proj_dim

        q3 = ops.reshape()(query_states, [BH, S, D])  # [BH, S, D]
        k3 = ops.reshape()(key_states, [BH, S, D])  # [BH, S, D]
        v3 = ops.reshape()(value_states, [BH, S, D])  # [BH, S, D]
        scores3 = ops.bmm_rcr()(q3, k3)
        scores = ops.reshape()(scores3, [batch_size, self.n_heads, S, S])
        if position_bias is None:
            key_length = key_states._attrs["shape"][-2]
            real_seq_length = query_states._attrs["shape"][-2]
            if not self.has_relative_attention_bias:
                position_bias = ops.full()(
                    (1, self.n_heads, seq_length, key_length),
                    fill_value=0.0,
                    dtype=scores.dtype(),
                )
            else:
                position_bias = ops.relative_attention_bias(
                    self.relative_attention_bias.weight.tensor(),
                    real_seq_length,
                    key_length,
                )()

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        scores += position_bias

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = ops.cast()(
            ops.softmax()(ops.cast()(scores, "float32"), dim=-1), scores.dtype()
        )
        attn_w3 = ops.reshape()(attn_weights, [BH, S, S])
        context3 = ops.bmm_rrr()(attn_w3, v3)

        attn_output = ops.reshape()(context3, [batch_size, self.n_heads, S, D])
        attn_output = ops.transpose()(attn_output, 1, 2)
        attn_output = ops.reshape()(attn_output, [batch_size, -1, self.inner_dim])
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
        dtype="float16",
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            layer_idx=layer_idx,
            dtype=dtype,
        )
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, dtype=dtype
        )
        self.dropout = nn.Dropout(config.dropout_rate, dtype=dtype)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
        dtype="float16",
    ):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                layer_idx=layer_idx,
                dtype=dtype,
            )
        )

        self.layer.append(T5LayerFF(config, dtype=dtype))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        output_attentions=False,
        return_dict=True,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[
            1:
        ]  # Keep self-attention outputs and relative position weights

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        outputs = (hidden_states,)

        return (
            outputs + attention_outputs
        )  # hidden-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, embed_tokens=None, dtype="float16"):
        super().__init__()

        self.embed_tokens = embed_tokens

        self.block = nn.ModuleList(
            [
                T5Block(
                    config,
                    has_relative_attention_bias=bool(i == 0),
                    layer_idx=i,
                    dtype=dtype,
                )
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask=None,
        encoder_hidden_states=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size = ops.size()(input_ids, 0)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ops.flatten()(input_ids))
            inputs_embeds = ops.reshape()(
                inputs_embeds, [batch_size, *inputs_embeds._attrs["shape"]]
            )

        encoder_extended_attention_mask = None

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = None
        encoder_decoder_position_bias = None
        position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                None,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                output_attentions=output_attentions,
                return_dict=return_dict,
            )

            hidden_states = layer_outputs[0]

            position_bias = layer_outputs[1]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class T5EncoderModel(nn.Module):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config, dtype="float16"):
        super().__init__()
        self.shared = nn.Embedding([config.vocab_size, config.d_model], dtype=dtype)

        encoder_config = config
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, dtype=dtype)

    def forward(
        self,
        input_ids: Annotated[
            Tensor, (Shape(name="batch_size"), Shape("sequence_length"))
        ],
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[tuple[Tensor], BaseModelOutput]:
        r"""
        input_ids (`Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else False

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
