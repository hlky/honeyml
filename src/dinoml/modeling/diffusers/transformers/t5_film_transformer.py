import math
from typing import Optional, Tuple

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..attention_processor import Attention
from ..embeddings import get_timestep_embedding, SiLU


class T5FilmDecoder(nn.Module):
    r"""
    T5 style decoder with FiLM conditioning.

    Args:
        input_dims (`int`, *optional*, defaults to `128`):
            The number of input dimensions.
        targets_length (`int`, *optional*, defaults to `256`):
            The length of the targets.
        d_model (`int`, *optional*, defaults to `768`):
            Size of the input hidden states.
        num_layers (`int`, *optional*, defaults to `12`):
            The number of `DecoderLayer`'s to use.
        num_heads (`int`, *optional*, defaults to `12`):
            The number of attention heads to use.
        d_kv (`int`, *optional*, defaults to `64`):
            Size of the key-value projection vectors.
        d_ff (`int`, *optional*, defaults to `2048`):
            The number of dimensions in the intermediate feed-forward layer of `DecoderLayer`'s.
        dropout_rate (`float`, *optional*, defaults to `0.1`):
            Dropout probability.
    """

    def __init__(
        self,
        input_dims: int = 128,
        targets_length: int = 256,
        max_decoder_noise_time: float = 2000.0,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_kv: int = 64,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        dtype: str = "float16",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_decoder_noise_time = max_decoder_noise_time
        self.dtype = dtype

        self.conditioning_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False, dtype=dtype),
            SiLU(),
            nn.Linear(d_model * 4, d_model * 4, bias=False, dtype=dtype),
            SiLU(),
        )

        self.position_encoding = nn.Embedding([targets_length, d_model], dtype=dtype)

        self.continuous_inputs_projection = nn.Linear(
            input_dims, d_model, bias=False, dtype=dtype
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.decoders = nn.ModuleList()
        for lyr_num in range(num_layers):
            # FiLM conditional T5 decoder
            lyr = DecoderLayer(
                d_model=d_model,
                d_kv=d_kv,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                dtype=dtype,
            )
            self.decoders.append(lyr)

        self.decoder_norm = T5LayerNorm(d_model, dtype=dtype)

        self.post_dropout = nn.Dropout(p=dropout_rate)
        self.spec_out = nn.Linear(d_model, input_dims, bias=False, dtype=dtype)

    def encoder_decoder_mask(self, query_input: Tensor, key_input: Tensor) -> Tensor:
        mask = ops.unsqueeze(-1)(query_input) * ops.unsqueeze(-2)(key_input)
        return ops.unsqueeze(-3)(mask)

    def forward(self, encodings_and_masks, decoder_input_tokens, decoder_noise_time):
        batch, _, _ = ops.size()(decoder_input_tokens)

        # decoder_noise_time is in [0, 1), so rescale to expected timing range.
        time_steps = ops.cast()(
            get_timestep_embedding(
                decoder_noise_time * self.max_decoder_noise_time,
                embedding_dim=self.d_model,
                max_period=self.max_decoder_noise_time,
            ),
            dtype=self.dtype,
        )

        conditioning_emb = ops.unsqueeze(1)(self.conditioning_emb(time_steps))

        seq_length = ops.size()(decoder_input_tokens, dim=1)

        # If we want to use relative positions for audio context, we can just offset
        # this sequence by the length of encodings_and_masks.
        decoder_positions = Tensor(
            [batch, seq_length], name="decoder_positions"
        )  # torch.arange(seq_length)

        position_encodings = self.position_encoding(decoder_positions)

        inputs = self.continuous_inputs_projection(decoder_input_tokens)
        inputs += position_encodings
        y = self.dropout(inputs)

        # decoder: No padding present.
        decoder_mask = ops.full()(
            shape=ops.size()(decoder_input_tokens)[:2],
            fill_value=1.0,
            dtype=inputs.dtype(),
        )

        # Translate encoding masks to encoder-decoder masks.
        encodings_and_encdec_masks = [
            (x, self.encoder_decoder_mask(decoder_mask, y))
            for x, y in encodings_and_masks
        ]

        # cross attend style: concat encodings
        encoded = ops.concatenate()([x[0] for x in encodings_and_encdec_masks], dim=1)
        encoder_decoder_mask = ops.concatenate()(
            [x[1] for x in encodings_and_encdec_masks], dim=-1
        )

        for lyr in self.decoders:
            y = lyr(
                y,
                conditioning_emb=conditioning_emb,
                encoder_hidden_states=encoded,
                encoder_attention_mask=encoder_decoder_mask,
            )[0]

        y = self.decoder_norm(y)
        y = self.post_dropout(y)

        spec_out = self.spec_out(y)
        return spec_out


class DecoderLayer(nn.Module):
    r"""
    T5 decoder layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`, *optional*, defaults to `1e-6`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float,
        layer_norm_epsilon: float = 1e-6,
        dtype: str = "float16",
    ):
        super().__init__()
        self.layer = nn.ModuleList()

        # cond self attention: layer 0
        self.layer.append(
            T5LayerSelfAttentionCond(
                d_model=d_model,
                d_kv=d_kv,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                dtype=dtype,
            )
        )

        # cross attention: layer 1
        self.layer.append(
            T5LayerCrossAttention(
                d_model=d_model,
                d_kv=d_kv,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
            )
        )

        # Film Cond MLP + dropout: last layer
        self.layer.append(
            T5LayerFFCond(
                d_model=d_model,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
            )
        )

    def forward(
        self,
        hidden_states: Tensor,
        conditioning_emb: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_decoder_position_bias=None,
    ) -> Tuple[Tensor]:
        hidden_states = self.layer[0](
            hidden_states,
            conditioning_emb=conditioning_emb,
            attention_mask=attention_mask,
        )

        if encoder_hidden_states is not None:
            encoder_extended_attention_mask = ops.cast()(
                ops.where()(encoder_attention_mask > 0, 0, -1e10),
                dtype=encoder_hidden_states.dtype(),
            )

            hidden_states = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_extended_attention_mask,
            )

        # Apply Film Conditional Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, conditioning_emb)

        return (hidden_states,)


class T5LayerSelfAttentionCond(nn.Module):
    r"""
    T5 style self-attention layer with conditioning.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        dropout_rate (`float`):
            Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float,
        dtype: str = "float16",
    ):
        super().__init__()
        self.layer_norm = T5LayerNorm(d_model, dtype=dtype)
        self.FiLMLayer = T5FiLMLayer(
            in_features=d_model * 4, out_features=d_model, dtype=dtype
        )
        self.attention = Attention(
            query_dim=d_model,
            heads=num_heads,
            dim_head=d_kv,
            out_bias=False,
            scale_qk=False,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        conditioning_emb: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # pre_self_attention_layer_norm
        normed_hidden_states = self.layer_norm(hidden_states)

        if conditioning_emb is not None:
            normed_hidden_states = self.FiLMLayer(
                normed_hidden_states, conditioning_emb
            )

        # Self-attention block
        attention_output = self.attention(normed_hidden_states)

        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states


class T5LayerCrossAttention(nn.Module):
    r"""
    T5 style cross-attention layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float,
        layer_norm_epsilon: float,
        dtype: str = "float16",
    ):
        super().__init__()
        self.attention = Attention(
            query_dim=d_model,
            heads=num_heads,
            dim_head=d_kv,
            out_bias=False,
            scale_qk=False,
            dtype=dtype,
        )
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            encoder_hidden_states=key_value_states,
            attention_mask=ops.squeeze(1)(attention_mask),
        )
        layer_output = hidden_states + self.dropout(attention_output)
        return layer_output


class T5LayerFFCond(nn.Module):
    r"""
    T5 style feed-forward conditional layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        layer_norm_epsilon: float,
        dtype: str = "float16",
    ):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, dtype=dtype
        )
        self.film = T5FiLMLayer(
            in_features=d_model * 4, out_features=d_model, dtype=dtype
        )
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, hidden_states: Tensor, conditioning_emb: Optional[Tensor] = None
    ) -> Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        if conditioning_emb is not None:
            forwarded_states = self.film(forwarded_states, conditioning_emb)

        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    r"""
    T5 style feed-forward layer with gated activations and dropout.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        dtype: str = "float16",
    ):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False, dtype=dtype)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False, dtype=dtype)
        self.wo = nn.Linear(d_ff, d_model, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = NewGELUActivation()

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerNorm(nn.Module):
    r"""
    T5 style layer normalization module.

    Args:
        hidden_size (`int`):
            Size of the input hidden states.
        eps (`float`, `optional`, defaults to `1e-6`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: str = "float16",
    ):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter([hidden_size], dtype=dtype)
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = ops.reduce_mean(dim=-1, keepdim=True)(
            ops.pow(ops.cast()(hidden_states, dtype="float32"), 2)
        )
        hidden_states = hidden_states * (
            1.0 / ops.sqrt(variance + self.variance_epsilon)
        )

        # convert into half-precision if necessary
        if self.weight.tensor().dtype() in ["float16", "bfloat16"]:
            hidden_states = ops.cast()(
                hidden_states, dtype=self.weight.tensor().dtype()
            )

        return self.weight.tensor() * hidden_states


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + ops.tanh(
                    math.sqrt(2.0 / math.pi) * (input + 0.044715 * ops.pow(input, 3.0))
                )
            )
        )


class T5FiLMLayer(nn.Module):
    """
    T5 style FiLM Layer.

    Args:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: str = "float16",
    ):
        super().__init__()
        self.scale_bias = nn.Linear(
            in_features, out_features * 2, bias=False, dtype=dtype
        )

    def forward(self, x: Tensor, conditioning_emb: Tensor) -> Tensor:
        emb = self.scale_bias(conditioning_emb)
        scale, shift = ops.chunk()(emb, 2, -1)
        x = x * (1 + scale) + shift
        return x
