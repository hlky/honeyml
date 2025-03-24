import math

from typing import Optional

import numpy as np
from honey.compiler import ops

from honey.frontend import IntVar, nn, Tensor

from .activations import FP32SiLU, get_activation

# TODO: sync with diffusers


def get_shape(x):
    shape = [
        (
            it.value()
            if not isinstance(it, IntVar)
            else [it.lower_bound(), it.upper_bound()]
        )
        for it in x._attrs["shape"]
    ]
    return shape


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        ops.arange(0, grid_size[0], 1)()
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        ops.arange(0, grid_size[1], 1)()
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = ops.meshgrid()(grid_w, grid_h)  # here w goes first
    grid = ops.stack()(grid, dim=0)

    grid = ops.reshape()(grid, [2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = ops.concatenate()(
            [ops.full()([extra_tokens, embed_dim], fill_value=0.0), pos_embed], dim=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    grid_0, grid_1 = ops.chunk()(grid, 2, dim=0)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_0)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_1)  # (H*W, D/2)

    emb = ops.concatenate()([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = ops.arange(0, embed_dim // 2, 1)()
    omega /= embed_dim / 2.0
    omega = 1.0 / ops.pow(10000, omega)  # (D/2,)

    pos = ops.reshape()(pos, [-1])  # (M,)
    pos = ops.reshape()(pos, [-1, 1])  # (M, 1)
    omega = ops.reshape()(omega, [1, -1])  # (1, D/2)

    out = ops.gemm_rrr()(pos, omega)  # (M, D/2) outer product

    emb_sin = ops.sin(out)  # (M, D/2)
    emb_cos = ops.cos(out)  # (M, D/2)

    emb = ops.concatenate()([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert timesteps._rank() == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    exponent = (-math.log(max_period)) * ops.arange(0, half_dim, 1)()

    exponent = exponent * (1.0 / (half_dim - downscale_freq_shift))

    emb = ops.cast()(ops.exp(exponent), timesteps.dtype())
    emb = ops.reshape()(timesteps, [-1, 1]) * ops.reshape()(emb, [1, -1])

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ops.concatenate()(
            [ops.cos(emb), ops.sin(emb)],
            dim=-1,
        )
    else:
        emb = ops.concatenate()(
            [ops.sin(emb), ops.cos(emb)],
            dim=-1,
        )
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
        dtype: str = "float16",
    ):
        super().__init__()

        self.linear_1 = nn.Linear(
            in_channels, time_embed_dim, sample_proj_bias, dtype=dtype
        )

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(
                cond_proj_dim, in_channels, bias=False, dtype=dtype
            )
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(
            time_embed_dim, time_embed_dim_out, sample_proj_bias, dtype=dtype
        )

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample: Tensor, condition: Optional[Tensor] = None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        dtype: str = "float16",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.dtype = dtype

    def forward(self, timesteps: Tensor):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim,
        size_emb_dim,
        use_additional_conditions: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype
        )

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            raise NotImplementedError("use_additional_conditions")
            self.additional_condition_proj = Timesteps(
                num_channels=256,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
                dtype=dtype,
            )
            self.additional_condition_proj_ar = Timesteps(
                num_channels=256,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
                dtype=dtype,
            )
            self.resolution_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=size_emb_dim, dtype=dtype
            )
            self.aspect_ratio_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=size_emb_dim, dtype=dtype
            )

    def forward(
        self,
        timestep: Tensor,
        resolution: Tensor,
        aspect_ratio: Tensor,
        batch_size=None,
    ):
        batch_size = ops.size()(timestep, dim=0)
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        if self.use_additional_conditions:
            assert (
                resolution is not None and aspect_ratio is not None
            ), "Additional conditions are required."
            resolution_emb = self.additional_condition_proj(ops.flatten()(resolution))
            resolution_emb = ops.reshape()(
                self.resolution_embedder(resolution_emb), [batch_size, -1]
            )
            aspect_ratio_emb = self.additional_condition_proj_ar(
                ops.flatten()(aspect_ratio)
            )
            aspect_ratio_emb = ops.reshape()(
                self.aspect_ratio_embedder(aspect_ratio_emb), [batch_size, -1]
            )
            emb = ops.concatenate()([resolution_emb, aspect_ratio_emb], dim=1)
            conditioning = timesteps_emb + emb
        else:
            conditioning = timesteps_emb

        return conditioning


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        layer_norm: bool = False,
        flatten: bool = True,
        bias: bool = True,
        interpolation_scale: float = 1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
        dtype: str = "float16",
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.dtype = dtype
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        if bias:
            self.proj = nn.Conv2dBias(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                dtype=dtype,
            )
        else:
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                dtype=dtype,
            )
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, elementwise_affine=False, eps=1e-6, dtype=dtype
            )
        else:
            self.norm = None

        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
            )
            self.pos_embed = ops.unsqueeze(0)(pos_embed)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height / self.patch_size
        width = width / self.patch_size
        if any([value > self.pos_embed_max_size for value in height._attrs["values"]]):
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if any([value > self.pos_embed_max_size for value in width._attrs["values"]]):
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) / 2
        left = (self.pos_embed_max_size - width) / 2
        spatial_pos_embed = ops.reshape()(
            self.pos_embed, [1, self.pos_embed_max_size, self.pos_embed_max_size, -1]
        )
        top_indices = ops.cast()(ops.arange(top, top + height, 1)(), "int64")
        left_indices = ops.cast()(ops.arange(left, left + width, 1)(), "int64")
        spatial_pos_embed = ops.index_select(dim=1)(spatial_pos_embed, top_indices)
        spatial_pos_embed = ops.index_select(dim=2)(spatial_pos_embed, left_indices)
        spatial_pos_embed = ops.reshape()(
            spatial_pos_embed, [1, -1, ops.size()(spatial_pos_embed, dim=-1)]
        )

        return spatial_pos_embed

    def forward(self, latent: Tensor):
        # Directly accessing shape rather than ops.size to keep the named IntVar
        height, width = (latent._attrs["shape"][1], latent._attrs["shape"][2])
        if self.pos_embed_max_size is None:
            height, width = (height / self.patch_size, width / self.patch_size)

        latent = self.proj(latent)
        if self.flatten:
            latent = ops.flatten(1, 2)(latent)
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=ops.size()(self.pos_embed, dim=-1)
                    ._attrs["int_var"]
                    .symbolic_value(),
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = ops.unsqueeze(0)(pos_embed)
            else:
                pos_embed = self.pos_embed

        pos_embed._attrs["shape"] = latent._attrs["shape"]
        return latent + pos_embed


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: Optional[int] = None,
        act_fn: str = "gelu_tanh",
        dtype: str = "float16",
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features, hidden_size, dtype=dtype)
        if act_fn == "gelu_tanh":
            self.act_1 = ops.fast_gelu
        elif act_fn == "silu":
            self.act_1 = ops.silu
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, caption: Tensor):
        hidden_states = self.linear_1(caption)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LabelEmbedding(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float,
        dtype: str = "float16",
    ):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            [num_classes + use_cfg_embedding, hidden_size], dtype=dtype
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        # NOTE: Honey workaround
        self.training = False

    def token_drop(self, labels: Tensor, force_drop_ids: Optional[Tensor] = None):
        """
        Drops labels to enable classifier-free guidance.
        """
        raise NotImplementedError("token_drop not yet implemented.")
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = torch.tensor(force_drop_ids == 1)
        labels = ops.where()(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: Tensor, force_drop_ids: Optional[Tensor] = None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(ops.flatten()(labels))
        return ops.unsqueeze(0)(embeddings)


class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        class_dropout_prob: float = 0.1,
        dtype: str = "float16",
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            dtype=dtype,
        )
        self.class_embedder = LabelEmbedding(
            num_classes,
            embedding_dim,
            class_dropout_prob,
            dtype=dtype,
        )

    def forward(self, timestep: Tensor, class_labels: Tensor, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_labels  # (N, D)

        return conditioning


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self,
        embedding_size: int = 256,
        scale: float = 1.0,
        set_W_to_weight: bool = True,
        log: bool = True,
        flip_sin_to_cos: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.weight = nn.Parameter([embedding_size], dtype=dtype)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            self.W = nn.Parameter([embedding_size], dtype=dtype)

            self.weight = self.W

    def forward(self, x: Tensor):
        if self.log:
            x = ops.log(x)

        x_proj = (
            ops.unsqueeze(1)(x) * ops.unsqueeze(1)(self.weight.tensor()) * 2 * math.pi
        )

        if self.flip_sin_to_cos:
            out = ops.concatenate()(
                [ops.cos(x_proj), ops.sin(x_proj)],
                dim=-1,
            )
        else:
            out = ops.concatenate()(
                [ops.sin(x_proj), ops.cos(x_proj)],
                dim=-1,
            )
        return out


class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(
        self, embed_dim: int, max_seq_length: int = 32, dtype: str = "float16"
    ):
        super().__init__()
        position = ops.unsqueeze(1)(
            Tensor([max_seq_length], name="position", dtype=dtype)
        )
        div_term = ops.exp(
            Tensor([embed_dim // 2], name="div_term", dtype=dtype)
            * (-math.log(10000.0) / embed_dim)
        )
        raise NotImplementedError("slice assignment not implemented.")
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


class ImagePositionalEmbeddings(nn.Module):
    """
    Converts latent image classes into vector embeddings. Sums the vector embeddings with positional embeddings for the
    height and width of the latent space.

    For more details, see figure 10 of the dall-e paper: https://arxiv.org/abs/2102.12092

    For VQ-diffusion:

    Output vector embeddings are used as input for the transformer.

    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    def __init__(
        self,
        num_embed: int,
        height: int,
        width: int,
        embed_dim: int,
        dtype: str = "float16",
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.emb = nn.Embedding([self.num_embed, embed_dim], dtype=dtype)
        self.height_emb = nn.Embedding([self.height, embed_dim], dtype=dtype)
        self.width_emb = nn.Embedding([self.width, embed_dim], dtype=dtype)
        self.dtype = dtype

    def forward(self, index: Tensor):
        emb = self.emb(index)

        height_emb = self.height_emb(
            ops.unsqueeze(0)(Tensor([self.height], name="height", dtype=self.dtype))
        )

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = ops.unsqueeze(2)(height_emb)

        width_emb = self.width_emb(
            ops.unsqueeze(0)(Tensor([self.width], name="width", dtype=self.dtype))
        )

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = ops.unsqueeze(1)(width_emb)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = ops.reshape()(pos_emb, [1, self.height * self.width, -1])

        emb = emb + ops.dynamic_slice()(
            pos_emb,
            [0, 0, 0],
            [None, ops.size()(emb, dim=1)._attrs["int_var"]._attrs["values"][0], None],
        )
        # emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


class TextImageProjection(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
        dtype: str = "float16",
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(
            image_embed_dim,
            self.num_image_text_embeds * cross_attention_dim,
            dtype=dtype,
        )
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim, dtype=dtype)

    def forward(self, text_embeds: Tensor, image_embeds: Tensor):
        batch_size = ops.size()(text_embeds, dim=0)

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = ops.reshape()(
            image_text_embeds, [batch_size, self.num_image_text_embeds, -1]
        )

        # text
        text_embeds = self.text_proj(text_embeds)

        return ops.concatenate()([image_text_embeds, text_embeds], dim=1)


class ImageProjection(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(
            image_embed_dim,
            self.num_image_text_embeds * cross_attention_dim,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(cross_attention_dim, dtype=dtype)

    def forward(self, image_embeds: Tensor):
        batch_size = ops.size()(image_embeds, dim=0)

        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = ops.reshape()(
            image_embeds, [batch_size, self.num_image_text_embeds, -1]
        )
        image_embeds = self.norm(image_embeds)
        return image_embeds


class TextTimeEmbedding(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        time_embed_dim: int,
        num_heads: int = 64,
        dtype: str = "float16",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(encoder_dim, dtype=dtype)
        self.pool = AttentionPooling(num_heads, encoder_dim, dtype=dtype)
        self.proj = nn.Linear(encoder_dim, time_embed_dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(time_embed_dim, dtype=dtype)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TextImageTimeEmbedding(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 768,
        image_embed_dim: int = 768,
        time_embed_dim: int = 1536,
        dtype: str = "float16",
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim, dtype=dtype)
        self.text_norm = nn.LayerNorm(time_embed_dim, dtype=dtype)
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim, dtype=dtype)

    def forward(self, text_embeds: Tensor, image_embeds: Tensor):
        # text
        time_text_embeds: Tensor = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # image
        time_image_embeds: Tensor = self.image_proj(image_embeds)

        return time_image_embeds + time_text_embeds


class ImageTimeEmbedding(nn.Module):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)

    def forward(self, image_embeds: Tensor):
        # image
        time_image_embeds: Tensor = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        return time_image_embeds


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = ops.silu

    def forward(self, x: Tensor):
        return self.op(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = ops.gelu

    def forward(self, x: Tensor):
        return self.op(x)


class ImageHintTimeEmbedding(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        time_embed_dim: int = 1536,
        dtype: str = "float16",
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)
        self.input_hint_block = nn.Sequential(
            nn.Conv2dBias(3, 16, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(16, 16, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(16, 32, 3, padding=1, stride=2, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(32, 32, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(32, 96, 3, padding=1, stride=2, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(96, 96, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(96, 256, 3, padding=1, stride=2, dtype=dtype),
            SiLU(),
            nn.Conv2dBias(256, 4, 3, padding=1, dtype=dtype),
        )

    def forward(self, image_embeds: Tensor, hint: Tensor):
        # image
        time_image_embeds: Tensor = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        hint = self.input_hint_block(hint)
        return time_image_embeds, hint


class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter([1, embed_dim], dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x: Tensor):
        raise NotImplementedError("TODO")
        bs, length, width = ops.size()(x)

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(
            x.dtype
        )
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = torch.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(
        self, embedding_dim: int, pooled_projection_dim: int, dtype: str = "float16"
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, dtype=dtype
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu", dtype=dtype
        )

    def forward(self, timestep: Tensor, pooled_projection: Tensor):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            ops.cast()(timesteps_proj, dtype=pooled_projection.dtype())
        )  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(
        self, embedding_dim: int, pooled_projection_dim: int, dtype: str = "float16"
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, dtype=dtype
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )

    def forward(self, timestep: Tensor, guidance: Tensor, pooled_projection: Tensor):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            ops.cast()(timesteps_proj, dtype=pooled_projection.dtype())
        )  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(
            ops.cast()(guidance_proj, dtype=pooled_projection.dtype())
        )  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning
