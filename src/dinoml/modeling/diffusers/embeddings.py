import math

from typing import Optional, Tuple, Union

import numpy as np
from dinoml.compiler import ops

from dinoml.frontend import IntVar, nn, Tensor
from dinoml.modeling.diffusers.attention_processor import Attention

from .activations import FP32SiLU, get_activation


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
    return ops.get_timestep_embedding()(
        timesteps=timesteps,
        embedding_dim=embedding_dim,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=max_period,
    )


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> Tensor:
    return ops.get_3d_sincos_pos_embed()(
        embed_dim=embed_dim,
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        spatial_interpolation_scale=spatial_interpolation_scale,
        temporal_interpolation_scale=temporal_interpolation_scale,
    )


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    return ops.get_2d_sincos_pos_embed()(
        embed_dim=embed_dim,
        grid_size=grid_size,
        cls_token=cls_token,
        extra_tokens=extra_tokens,
        interpolation_scale=interpolation_scale,
        base_size=base_size,
    )


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

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,
            bias=bias,
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
        self.embed_dim = embed_dim
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
        return ops.cropped_pos_embed()(
            embed_dim=self.embed_dim,
            pos_embed_max_size=self.pos_embed_max_size,
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
            patch_size=self.patch_size,
            height=height,
            width=width,
        )

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
                    embed_dim=self.embed_dim,
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = ops.unsqueeze(0)(pos_embed)
            else:
                pos_embed = self.pos_embed

        return latent + pos_embed


class LuminaPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for Lumina-T2X

    Args:
        patch_size (`int`, defaults to `2`): The size of the patches.
        in_channels (`int`, defaults to `4`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=embed_dim,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x: Tensor, freqs_cis: Tensor):
        """
        Patchifies and embeds the input tensor(s).

        Args:
            x (List[Tensor] | Tensor): The input tensor(s) to be patchified and embedded.

        Returns:
            Tuple[Tensor, Tensor, List[Tuple[int, int]], Tensor]: A tuple containing the patchified
            and embedded tensor(s), the mask indicating the valid patches, the original image size(s), and the
            frequency tensor(s).
        """
        patch_height = patch_width = self.patch_size
        batch_size, height, width, channel = x._attrs["shape"]
        height_tokens, width_tokens = height / patch_height, width / patch_width

        x = ops.reshape()(
            x,
            batch_size,
            height_tokens,
            patch_height,
            width_tokens,
            patch_width,
            channel,
        )
        x = ops.permute()(x, [0, 1, 3, 5, 2, 4])
        x = ops.flatten(end_dim=3)(x)
        x = self.proj(x)
        x = ops.flatten(1, 2)(x)

        mask = ops.full()(
            [x._attrs["shape"][0], x._attrs["shape"][1]], fill_value=1, dtype="int"
        )

        return (
            x,
            mask,
            [(height, width)] * batch_size,
            freqs_cis[:height_tokens, :width_tokens].flatten(0, 1).unsqueeze(0),
        )


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
        dtype: str = "float16",
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=patch_size,
                bias=bias,
                dtype=dtype,
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.proj = nn.Linear(
                in_channels * patch_size * patch_size * patch_size_t,
                embed_dim,
                dtype=dtype,
            )

        self.text_proj = nn.Linear(text_embed_dim, embed_dim, dtype=dtype)

        if use_positional_embeddings:
            self.pos_embedding = self._get_positional_embeddings(
                sample_height, sample_width, sample_frames
            )
        elif use_learned_positional_embeddings:
            # `THUDM/CogVideoX-5b-I2V`
            self.pos_embedding = nn.Parameter([1, 17776, 3072], dtype=dtype)
            # TODO: check if register buffer works
            # self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(
        self, sample_height: int, sample_width: int, sample_frames: int
    ) -> Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (
            sample_frames - 1
        ) // self.temporal_compression_ratio + 1
        num_patches = (
            post_patch_height * post_patch_width * post_time_compression_frames
        )

        joint_pos_embedding = ops.get_3d_sincos_pos_embed_cogvideox()(
            embed_dim=self.embed_dim,
            spatial_size=(post_patch_width, post_patch_height),
            temporal_size=post_time_compression_frames,
            max_text_seq_length=self.max_text_seq_length,
            spatial_interpolation_scale=self.spatial_interpolation_scale,
            temporal_interpolation_scale=self.temporal_interpolation_scale,
        )

        return joint_pos_embedding

    def forward(self, text_embeds: Tensor, image_embeds: Tensor):
        r"""
        Args:
            text_embeds (`Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, height, width, channels).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, height, width, channels = image_embeds._attrs["shape"]

        if self.patch_size_t is None:
            image_embeds = ops.reshape()(image_embeds, [-1, height, width, channels])
            image_embeds = self.proj(image_embeds)
            image_embeds = ops.reshape()(
                image_embeds,
                [batch_size, num_frames, *image_embeds._attrs["shape"][1:]],
            )
            image_embeds = ops.reshape()(
                image_embeds, [batch_size, num_frames, -1, channels]
            )
            image_embeds = ops.reshape()(image_embeds, [batch_size, -1, channels])
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            # batch_size, num_frames, height, width, channels
            image_embeds = ops.reshape()(
                image_embeds,
                [
                    batch_size,
                    num_frames / p_t,
                    p_t,
                    height / p,
                    p,
                    width / p,
                    p,
                    channels,
                ],
            )
            # batch_size, num_frames / p_t, p_t, height / p, p, width / p, p, channels
            # batch_size, num_frames / p_t, height / p, width / p, channels, p_t, p, p
            image_embeds = ops.permute()(image_embeds, [0, 1, 3, 5, 7, 2, 4, 6])
            image_embeds = ops.flatten(4, 7)(image_embeds)
            image_embeds = ops.flatten(1, 3)(image_embeds)
            image_embeds = self.proj(image_embeds)

        embeds = ops.concatenate()(
            [text_embeds, image_embeds], dim=1
        )  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (
                self.sample_width != width or self.sample_height != height
            ):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                )

            pre_time_compression_frames = (
                num_frames - 1
            ) * self.temporal_compression_ratio + 1

            if (
                self.use_positional_embeddings
                # self.sample_height != height
                # or self.sample_width != width
                # or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(
                    height, width, pre_time_compression_frames
                )
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = ops.cast()(pos_embedding, dtype=embeds.dtype())
            embeds = embeds + pos_embedding

        return embeds


class CogView3PlusPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
        pos_embed_max_size: int = 128,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.pos_embed_max_size = pos_embed_max_size
        # Linear projection for image patches
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size, dtype=dtype)

        # Linear projection for text embeddings
        self.text_proj = nn.Linear(text_hidden_size, hidden_size, dtype=dtype)

        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, pos_embed_max_size, base_size=pos_embed_max_size
        )
        self.pos_embed = pos_embed

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        batch_size, height, width, channel = hidden_states._attrs["shape"]

        height = height / self.patch_size
        width = width / self.patch_size
        hidden_states = ops.reshape()(
            hidden_states,
            [batch_size, height, self.patch_size, width, self.patch_size, channel],
        )
        hidden_states = ops.permute()(hidden_states, [0, 1, 3, 5, 2, 4])
        hidden_states = ops.reshape()(
            hidden_states,
            [batch_size, height * width, channel * self.patch_size * self.patch_size],
        )

        # Project the patches
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        hidden_states = ops.concatenate()([encoder_hidden_states, hidden_states], dim=1)

        # Calculate text_length
        text_length = encoder_hidden_states._attrs["shape"][1]

        pos_embed = ops.get_2d_sincos_pos_embed_cogview3plus()(
            pos_table=self.pos_embed,
            hidden_size=self.hidden_size,
            pos_embed_max_size=self.pos_embed_max_size,
            height=height,
            width=width,
            text_length=text_length,
        )

        return hidden_states + ops.cast()(pos_embed, hidden_states.dtype())


def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    return ops.get_3d_rotary_pos_embed()(
        embed_dim=embed_dim,
        crops_coords=crops_coords,
        grid_size=grid_size,
        temporal_size=temporal_size,
        theta=theta,
        use_real=use_real,
        grid_type=grid_type,
        max_size=max_size,
    )


def get_3d_rotary_pos_embed_allegro(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    interpolation_scale_h: float = 2.0,
    interpolation_scale_t: float = 2.2,
    interpolation_scale_w: float = 2.0,
    attention_head_dim: int = 96,
):
    return ops.get_3d_rotary_pos_embed_allegro()(
        height=height,
        width=width,
        num_frames=num_frames,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        patch_size=patch_size,
        interpolation_scale_h=interpolation_scale_h,
        interpolation_scale_t=interpolation_scale_t,
        interpolation_scale_w=interpolation_scale_w,
        attention_head_dim=attention_head_dim,
    )


def get_2d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    use_real=True,
):
    """
    RoPE for image tokens with 2d structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size
    crops_coords (`Tuple[int]`)
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.
    device: (`torch.device`, **optional**):
        The device used to create tensors.

    Returns:
        `Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    return ops.get_2d_rotary_pos_embed()(
        embed_dim=embed_dim,
        crops_coords=crops_coords,
        grid_size=grid_size,
        use_real=use_real,
    )


def get_2d_rotary_pos_embed_lumina(
    embed_dim, len_h, len_w, linear_factor=1.0, ntk_factor=1.0
):
    return ops.get_2d_rotary_pos_embed_lumina()(
        embed_dim=embed_dim,
        len_h=len_h,
        len_w=len_w,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
    )


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
):
    return ops.get_1d_rotary_pos_embed()(
        dim=dim,
        pos=pos,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
    )


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


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self,
        embedding_size: int = 256,
        scale: float = 1.0,
        set_W_to_weight=True,
        log=True,
        flip_sin_to_cos=False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.weight = nn.Parameter([embedding_size], dtype=dtype)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            del self.weight
            # NOTE: this may still need mapping
            self.W = nn.Parameter([embedding_size], dtype=dtype)
            self.weight = self.W
            del self.W

    def forward(self, x: Tensor):
        return ops.gaussian_fourier_projection()(
            x, self.weight.tensor(), self.log, self.flip_sin_to_cos
        )


class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length

    def forward(self, x):
        return ops.sinusoidal_positional_embedding()(
            x, self.embed_dim, self.max_seq_length
        )


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
        # NOTE: DinoML workaround
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


class IPAdapterFullImageProjection(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 1024,
        cross_attention_dim: int = 1024,
        dtype: str = "float16",
    ):
        super().__init__()
        from .attention import FeedForward

        self.ff = FeedForward(
            image_embed_dim,
            cross_attention_dim,
            mult=1,
            activation_fn="gelu",
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(cross_attention_dim, dtype=dtype)

    def forward(self, image_embeds: Tensor) -> Tensor:
        return self.norm(self.ff(image_embeds))


class IPAdapterFaceIDImageProjection(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 1024,
        cross_attention_dim: int = 1024,
        mult: int = 1,
        num_tokens: int = 1,
        dtype: str = "float16",
    ):
        super().__init__()
        from .attention import FeedForward

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.ff = FeedForward(
            image_embed_dim,
            cross_attention_dim * num_tokens,
            mult=mult,
            activation_fn="gelu",
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(cross_attention_dim, dtype=dtype)

    def forward(self, image_embeds: Tensor) -> Tensor:
        x = self.ff(image_embeds)
        x = ops.reshape()(x, [-1, self.num_tokens, self.cross_attention_dim])
        return self.norm(x)


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

    def forward(self, timestep: Tensor, pooled_projection: Tensor) -> Tensor:
        timesteps_proj: Tensor = self.time_proj(timestep)
        timesteps_emb: Tensor = self.timestep_embedder(
            ops.cast()(timesteps_proj, dtype=pooled_projection.dtype())
        )  # (N, D)

        pooled_projections: Tensor = self.text_embedder(pooled_projection)

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

    def forward(
        self, timestep: Tensor, guidance: Tensor, pooled_projection: Tensor
    ) -> Tensor:
        timesteps_proj: Tensor = self.time_proj(timestep)
        timesteps_emb: Tensor = self.timestep_embedder(
            ops.cast()(timesteps_proj, dtype=pooled_projection.dtype())
        )  # (N, D)

        guidance_proj: Tensor = self.time_proj(guidance)
        guidance_emb: Tensor = self.guidance_embedder(
            ops.cast()(guidance_proj, dtype=pooled_projection.dtype())
        )  # (N, D)

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections: Tensor = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning


class CogView3CombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        pooled_projection_dim: int,
        timesteps_dim: int = 256,
        dtype: str = "float16",
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=timesteps_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            dtype=dtype,
        )
        self.condition_proj = Timesteps(
            num_channels=condition_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=timesteps_dim, time_embed_dim=embedding_dim, dtype=dtype
        )
        self.condition_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu", dtype=dtype
        )
        self.dtype = dtype

    def forward(
        self,
        timestep: Tensor,
        original_size: Tensor,
        target_size: Tensor,
        crop_coords: Tensor,
    ) -> Tensor:
        timesteps_proj: Tensor = self.time_proj(timestep)

        original_size_proj = ops.reshape()(
            self.condition_proj(ops.flatten()(original_size)),
            [original_size._attrs["shape"][0], -1],
        )
        crop_coords_proj = ops.reshape()(
            self.condition_proj(ops.flatten()(crop_coords)),
            [crop_coords._attrs["shape"][0], -1],
        )
        target_size_proj = ops.reshape()(
            self.condition_proj(ops.flatten()(target_size)),
            [target_size._attrs["shape"][0], -1],
        )

        # (B, 3 * condition_dim)
        condition_proj = ops.concatenate()(
            [original_size_proj, crop_coords_proj, target_size_proj], dim=1
        )

        timesteps_emb: Tensor = self.timestep_embedder(
            ops.cast()(timesteps_proj, dtype=self.dtype)
        )  # (B, embedding_dim)
        condition_emb: Tensor = self.condition_embedder(
            ops.cast()(condition_proj, dtype=self.dtype)
        )  # (B, embedding_dim)

        conditioning = timesteps_emb + condition_emb
        return conditioning


class HunyuanDiTAttentionPool(nn.Module):
    # Copied from https://github.com/Tencent/HunyuanDiT/blob/cb709308d92e6c7e8d59d0dff41b74d35088db6a/hydit/modules/poolers.py#L6

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            [spacial_dim + 1, embed_dim], dtype=dtype
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, dtype=dtype)
        self.num_heads = num_heads

    def forward(self, x: Tensor):
        raise NotImplementedError(f"{__class__.__name__}")
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        pooled_projection_dim=1024,
        seq_len=256,
        cross_attention_dim=2048,
        use_style_cond_and_image_meta_size=True,
        dtype: str = "float16",
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, dtype=dtype
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype
        )

        self.size_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, dtype=dtype
        )

        self.pooler = HunyuanDiTAttentionPool(
            seq_len,
            cross_attention_dim,
            num_heads=8,
            output_dim=pooled_projection_dim,
            dtype=dtype,
        )

        # Here we use a default learned embedder layer for future extension.
        self.use_style_cond_and_image_meta_size = use_style_cond_and_image_meta_size
        if use_style_cond_and_image_meta_size:
            self.style_embedder = nn.Embedding([1, embedding_dim], dtype=dtype)
            extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        else:
            extra_in_dim = pooled_projection_dim

        self.extra_embedder = PixArtAlphaTextProjection(
            in_features=extra_in_dim,
            hidden_size=embedding_dim * 4,
            out_features=embedding_dim,
            act_fn="silu_fp32",
            dtype=dtype,
        )
        self.dtype = dtype

    def forward(
        self,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        image_meta_size: Tensor,
        style: Tensor,
    ):
        timesteps_proj: Tensor = self.time_proj(timestep)
        timesteps_emb: Tensor = self.timestep_embedder(
            ops.cast()(timesteps_proj, dtype=self.dtype)
        )  # (N, 256)

        # extra condition1: text
        pooled_projections: Tensor = self.pooler(encoder_hidden_states)  # (N, 1024)

        if self.use_style_cond_and_image_meta_size:
            # extra condition2: image meta size embedding
            image_meta_size = self.size_proj(ops.reshape()(image_meta_size, [-1]))
            image_meta_size = ops.cast()(image_meta_size, dtype=self.dtype)
            image_meta_size = ops.reshape()(image_meta_size, [-1, 6 * 256])  # (N, 1536)

            # extra condition3: style embedding
            style_embedding: Tensor = self.style_embedder(style)  # (N, embedding_dim)

            # Concatenate all extra vectors
            extra_cond = ops.concatenate()(
                [pooled_projections, image_meta_size, style_embedding], dim=1
            )
        else:
            extra_cond = ops.concatenate()([pooled_projections], dim=1)

        conditioning: Tensor = timesteps_emb + self.extra_embedder(extra_cond)  # [B, D]

        return conditioning


class LuminaCombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        cross_attention_dim: int = 2048,
        frequency_embedding_size: int = 256,
        dtype: str = "float16",
    ):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            dtype=dtype,
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size,
            time_embed_dim=hidden_size,
            dtype=dtype,
        )

        self.caption_embedder = nn.Sequential(
            nn.LayerNorm(cross_attention_dim, dtype=dtype),
            nn.Linear(cross_attention_dim, hidden_size, bias=True, dtype=dtype),
        )

    def forward(
        self, timestep: Tensor, caption_feat: Tensor, caption_mask: Tensor
    ) -> Tensor:
        # timestep embedding:
        time_freq: Tensor = self.time_proj(timestep)
        time_embed: Tensor = self.timestep_embedder(
            ops.cast()(time_freq, dtype=caption_feat.dtype())
        )

        # caption condition embedding:
        caption_mask_float = ops.unsqueeze(-1)(
            ops.cast()(caption_mask, dtype="float32")
        )
        caption_feats_pool = ops.reduce_sum(dim=1)(
            caption_feat * caption_mask_float
        ) / ops.reduce_sum(dim=1)(caption_mask_float)
        caption_feats_pool = ops.cast()(caption_feats_pool, dtype=caption_feat.dtype())
        caption_embed: Tensor = self.caption_embedder(caption_feats_pool)

        conditioning = time_embed + caption_embed

        return conditioning


class MochiCombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        text_embed_dim: int,
        time_embed_dim: int = 256,
        num_attention_heads: int = 8,
        dtype: str = "float16",
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=time_embed_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=time_embed_dim, time_embed_dim=embedding_dim, dtype=dtype
        )
        self.pooler = MochiAttentionPool(
            num_attention_heads=num_attention_heads,
            embed_dim=text_embed_dim,
            output_dim=embedding_dim,
            dtype=dtype,
        )
        self.caption_proj = nn.Linear(
            text_embed_dim, pooled_projection_dim, dtype=dtype
        )
        self.dtype = dtype

    def forward(
        self,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        time_proj: Tensor = self.time_proj(timestep)
        time_emb: Tensor = self.timestep_embedder(
            ops.cast()(time_proj, dtype=self.dtype)
        )

        pooled_projections: Tensor = self.pooler(
            encoder_hidden_states, encoder_attention_mask
        )
        caption_proj: Tensor = self.caption_proj(encoder_hidden_states)

        conditioning = time_emb + pooled_projections
        return conditioning, caption_proj


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

    def forward(self, hidden_states: Tensor) -> Tensor:
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

    def forward(self, text_embeds: Tensor, image_embeds: Tensor) -> Tensor:
        # text
        time_text_embeds: Tensor = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # image
        time_image_embeds: Tensor = self.image_proj(image_embeds)

        return time_image_embeds + time_text_embeds


class ImageTimeEmbedding(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        time_embed_dim: int = 1536,
        dtype: str = "float16",
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim, dtype=dtype)
        self.image_norm = nn.LayerNorm(time_embed_dim, dtype=dtype)

    def forward(self, image_embeds: Tensor) -> Tensor:
        # image
        time_image_embeds: Tensor = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        return time_image_embeds


class ImageHintTimeEmbedding(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        time_embed_dim: int = 1536,
        dtype: str = "float16",
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim, dtype=dtype)
        self.image_norm = nn.LayerNorm(time_embed_dim, dtype=dtype)
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2, dtype=dtype),
            SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2, dtype=dtype),
            SiLU(),
            nn.Conv2d(96, 96, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2, dtype=dtype),
            SiLU(),
            nn.Conv2d(256, 4, 3, padding=1, dtype=dtype),
        )

    def forward(self, image_embeds: Tensor, hint: Tensor) -> Tuple[Tensor, Tensor]:
        # image
        time_image_embeds: Tensor = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        hint = self.input_hint_block(hint)
        return time_image_embeds, hint


class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads: int, embed_dim: int, dtype: str = "float16"):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter([1, embed_dim], dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x: Tensor):
        raise NotImplementedError(f"{__class__.__name__}")
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


class MochiAttentionPool(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        embed_dim: int,
        output_dim: Optional[int] = None,
        dtype: str = "float16",
    ) -> None:
        super().__init__()

        self.output_dim = output_dim or embed_dim
        self.num_attention_heads = num_attention_heads

        self.to_kv = nn.Linear(embed_dim, 2 * embed_dim, dtype=dtype)
        self.to_q = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.to_out = nn.Linear(embed_dim, self.output_dim, dtype=dtype)

    @staticmethod
    def pool_tokens(x: Tensor, mask: Tensor, *, keepdim=False) -> Tensor:
        """
        Pool tokens in x using mask.

        NOTE: We assume x does not require gradients.

        Args:
            x: (B, L, D) tensor of tokens.
            mask: (B, L) boolean tensor indicating which tokens are not padding.

        Returns:
            pooled: (B, D) tensor of pooled tokens.
        """
        assert x.size(1) == mask.size(1)  # Expected mask to have same length as tokens.
        assert x.size(0) == mask.size(
            0
        )  # Expected mask to have same batch size as tokens.
        mask = mask[:, :, None].to(dtype=x.dtype)
        mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask).sum(dim=1, keepdim=keepdim)
        return pooled

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError(f"{__class__.__name__}")
        r"""
        Args:
            x (`Tensor`):
                Tensor of shape `(B, S, D)` of input tokens.
            mask (`Tensor`):
                Boolean ensor of shape `(B, S)` indicating which tokens are not padding.

        Returns:
            `Tensor`:
                `(B, D)` tensor of pooled tokens.
        """
        D = x.size(2)

        # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
        attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
        attn_mask = F.pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

        # Average non-padding token features. These will be used as the query.
        x_pool = self.pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

        # Concat pooled features to input sequence.
        x = torch.cat([x_pool, x], dim=1)  # (B, L+1, D)

        # Compute queries, keys, values. Only the mean token is used to create a query.
        kv = self.to_kv(x)  # (B, L+1, 2 * D)
        q = self.to_q(x[:, 0])  # (B, D)

        # Extract heads.
        head_dim = D // self.num_attention_heads
        kv = kv.unflatten(
            2, (2, self.num_attention_heads, head_dim)
        )  # (B, 1+L, 2, H, head_dim)
        kv = kv.transpose(1, 3)  # (B, H, 2, 1+L, head_dim)
        k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
        q = q.unflatten(1, (self.num_attention_heads, head_dim))  # (B, H, head_dim)
        q = q.unsqueeze(2)  # (B, H, 1, head_dim)

        # Compute attention.
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )  # (B, H, 1, head_dim)

        # Concatenate heads and run output.
        x = x.squeeze(2).flatten(1, 2)  # (B, D = H * head_dim)
        x = self.to_out(x)
        return x


def get_fourier_embeds_from_boundingbox(embed_dim: int, box: Tensor) -> Tensor:
    return ops.get_fourier_embeds_from_boundingbox()(embed_dim=embed_dim, box=box)


class GLIGENTextBoundingboxProjection(nn.Module):
    def __init__(
        self,
        positive_len: int,
        out_dim: int,
        feature_type: str = "text-only",
        fourier_freqs: int = 8,
        dtype: str = "float16",
    ):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        if feature_type == "text-only":
            self.linears = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512, dtype=dtype),
                SiLU(),
                nn.Linear(512, 512, dtype=dtype),
                SiLU(),
                nn.Linear(512, out_dim, dtype=dtype),
            )
            self.null_positive_feature = nn.Parameter([self.positive_len], dtype=dtype)

        elif feature_type == "text-image":
            self.linears_text = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512, dtype=dtype),
                SiLU(),
                nn.Linear(512, 512, dtype=dtype),
                SiLU(),
                nn.Linear(512, out_dim, dtype=dtype),
            )
            self.linears_image = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512, dtype=dtype),
                SiLU(),
                nn.Linear(512, 512, dtype=dtype),
                SiLU(),
                nn.Linear(512, out_dim, dtype=dtype),
            )
            self.null_text_feature = nn.Parameter([self.positive_len], dtype=dtype)
            self.null_image_feature = nn.Parameter([self.positive_len], dtype=dtype)

        self.null_position_feature = nn.Parameter([self.position_dim], dtype=dtype)

    def forward(
        self,
        boxes: Tensor,
        masks: Tensor,
        positive_embeddings: Optional[Tensor] = None,
        phrases_masks: Optional[Tensor] = None,
        image_masks: Optional[Tensor] = None,
        phrases_embeddings: Optional[Tensor] = None,
        image_embeddings: Optional[Tensor] = None,
    ):
        masks = ops.unsqueeze(-1)(masks)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(
            self.fourier_embedder_dim, boxes
        )  # B*N*4 -> B*N*C

        # learnable null embedding
        xyxy_null = ops.reshape()(self.null_position_feature, [1, 1, -1])

        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        # positionet with text only information
        if positive_embeddings is not None:
            # learnable null embedding
            positive_null = ops.reshape()(self.null_positive_feature, [1, 1, -1])

            # replace padding with learnable null embedding
            positive_embeddings = (
                positive_embeddings * masks + (1 - masks) * positive_null
            )

            objs = self.linears(
                ops.concatenate()([positive_embeddings, xyxy_embedding], dim=-1)
            )

        # positionet with text and image information
        else:
            phrases_masks = ops.unsqueeze(-1)(phrases_masks)
            image_masks = ops.unsqueeze(-1)(image_masks)

            # learnable null embedding
            text_null = ops.reshape()(self.null_text_feature, [1, 1, -1])
            image_null = ops.reshape()(self.null_image_feature, [1, 1, -1])

            # replace padding with learnable null embedding
            phrases_embeddings = (
                phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
            )
            image_embeddings = (
                image_embeddings * image_masks + (1 - image_masks) * image_null
            )

            objs_text = self.linears_text(
                ops.concatenate()([phrases_embeddings, xyxy_embedding], dim=-1)
            )
            objs_image = self.linears_image(
                ops.concatenate()([image_embeddings, xyxy_embedding], dim=-1)
            )
            objs = ops.concatenate()([objs_text, objs_image], dim=1)

        return objs


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
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
    ) -> Tensor:
        batch_size: IntVar = timestep._attrs["shape"][0]
        timesteps_proj: Tensor = self.time_proj(timestep)
        timesteps_emb: Tensor = self.timestep_embedder(timesteps_proj)  # (N, D)

        if self.use_additional_conditions:
            assert resolution is not None and aspect_ratio is not None, (
                "Additional conditions are required."
            )
            resolution_emb: Tensor = self.additional_condition_proj(
                ops.flatten()(resolution)
            )
            resolution_emb = ops.reshape()(
                self.resolution_embedder(resolution_emb), [batch_size, -1]
            )
            aspect_ratio_emb: Tensor = self.additional_condition_proj_ar(
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

    def forward(self, caption: Tensor) -> Tensor:
        hidden_states: Tensor = self.linear_1(caption)
        hidden_states: Tensor = self.linear_2(hidden_states)
        return hidden_states


class IPAdapterPlusImageProjectionBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int = 768,
        dim_head: int = 64,
        heads: int = 16,
        ffn_ratio: float = 4,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        from .attention import FeedForward

        self.ln0 = nn.LayerNorm(embed_dims, dtype=dtype)
        self.ln1 = nn.LayerNorm(embed_dims, dtype=dtype)
        self.attn = Attention(
            query_dim=embed_dims,
            dim_head=dim_head,
            heads=heads,
            out_bias=False,
            dtype=dtype,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dims, dtype=dtype),
            FeedForward(
                embed_dims,
                embed_dims,
                activation_fn="gelu",
                mult=ffn_ratio,
                bias=False,
                dtype=dtype,
            ),
        )

    def forward(self, x: Tensor, latents: Tensor, residual: Tensor) -> Tensor:
        encoder_hidden_states = self.ln0(x)
        latents = self.ln1(latents)
        encoder_hidden_states = ops.concatenate()(
            [encoder_hidden_states, latents], dim=-2
        )
        latents = self.attn(latents, encoder_hidden_states) + residual
        latents = self.ff(latents) + latents
        return latents


class IPAdapterPlusImageProjection(nn.Module):
    """Resampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_queries (int):
            The number of queries. Defaults to 8. ffn_ratio (float): The expansion ratio
        of feedforward network hidden
            layer channels. Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 1024,
        hidden_dims: int = 1280,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        ffn_ratio: float = 4,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter([1, num_queries, hidden_dims], dtype=dtype)

        self.proj_in = nn.Linear(embed_dims, hidden_dims, dtype=dtype)

        self.proj_out = nn.Linear(hidden_dims, output_dims, dtype=dtype)
        self.norm_out = nn.LayerNorm(output_dims, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                IPAdapterPlusImageProjectionBlock(
                    hidden_dims, dim_head, heads, ffn_ratio, dtype=dtype
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input Tensor.
        Returns:
            Tensor: Output Tensor.
        """
        latents = self.latents.tensor()

        x = self.proj_in(x)

        for block in self.layers:
            residual: Tensor = latents
            latents: Tensor = block(x, latents, residual)

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class IPAdapterFaceIDPlusImageProjection(nn.Module):
    """FacePerceiverResampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_tokens (int): Number of tokens num_queries (int): The number of queries. Defaults to 8.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        ffproj_ratio (float): The expansion ratio of feedforward network hidden
            layer channels (for ID embeddings). Defaults to 4.
    """

    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 768,
        hidden_dims: int = 1280,
        id_embeddings_dim: int = 512,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_tokens: int = 4,
        num_queries: int = 8,
        ffn_ratio: float = 4,
        ffproj_ratio: int = 2,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        from .attention import FeedForward

        self.num_tokens = num_tokens
        self.embed_dim = embed_dims
        self.clip_embeds = None
        self.shortcut = False
        self.shortcut_scale = 1.0

        self.proj = FeedForward(
            id_embeddings_dim,
            embed_dims * num_tokens,
            activation_fn="gelu",
            mult=ffproj_ratio,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(embed_dims, dtype=dtype)

        self.proj_in = nn.Linear(hidden_dims, embed_dims, dtype=dtype)

        self.proj_out = nn.Linear(embed_dims, output_dims, dtype=dtype)
        self.norm_out = nn.LayerNorm(output_dims, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                IPAdapterPlusImageProjectionBlock(
                    embed_dims, dim_head, heads, ffn_ratio, dtype=dtype
                )
                for _ in range(depth)
            ]
        )

    def forward(self, id_embeds: Tensor, clip_embeds: Tensor) -> Tensor:
        """Forward pass.

        Args:
            id_embeds (Tensor): Input Tensor (ID embeds).
        Returns:
            Tensor: Output Tensor.
        """
        id_embeds = ops.cast()(id_embeds, dtype=clip_embeds.dtype())
        id_embeds = self.proj(id_embeds)
        id_embeds = ops.reshape()(id_embeds, [-1, self.num_tokens, self.embed_dim])
        id_embeds = self.norm(id_embeds)
        latents = id_embeds

        clip_embeds = self.proj_in(clip_embeds)
        x = ops.reshape()(
            clip_embeds,
            [-1, clip_embeds._attrs["shape"][2], clip_embeds._attrs[".shape"][3]],
        )

        for block in self.layers:
            residual: Tensor = latents
            latents: Tensor = block(x, latents, residual)

        latents = self.proj_out(latents)
        out: Tensor = self.norm_out(latents)
        if self.shortcut:
            out = id_embeds + self.shortcut_scale * out
        return out


class IPAdapterTimeImageProjectionBlock(nn.Module):
    """Block for IPAdapterTimeImageProjection.

    Args:
        hidden_dim (`int`, defaults to 1280):
            The number of hidden channels.
        dim_head (`int`, defaults to 64):
            The number of head channels.
        heads (`int`, defaults to 20):
            Parallel attention heads.
        ffn_ratio (`int`, defaults to 4):
            The expansion ratio of feedforward network hidden layer channels.
    """

    def __init__(
        self,
        hidden_dim: int = 1280,
        dim_head: int = 64,
        heads: int = 20,
        ffn_ratio: int = 4,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        from .attention import FeedForward

        self.ln0 = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.attn = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=hidden_dim,
            dim_head=dim_head,
            heads=heads,
            bias=False,
            out_bias=False,
        )
        self.ff = FeedForward(
            hidden_dim,
            hidden_dim,
            activation_fn="gelu",
            mult=ffn_ratio,
            bias=False,
            dtype=dtype,
        )

        # AdaLayerNorm
        self.adaln_silu = SiLU()
        self.adaln_proj = nn.Linear(hidden_dim, 4 * hidden_dim, dtype=dtype)
        self.adaln_norm = nn.LayerNorm(hidden_dim, dtype=dtype)

        # Set attention scale and fuse KV
        self.attn.scale = 1 / math.sqrt(math.sqrt(dim_head))
        raise NotImplementedError(f"{__class__.__name__} Attention.fuse_projections")
        self.attn.fuse_projections()
        self.attn.to_k = None
        self.attn.to_v = None

    def forward(self, x: Tensor, latents: Tensor, timestep_emb: Tensor) -> Tensor:
        raise NotImplementedError(f"{__class__.__name__}")
        """Forward pass.

        Args:
            x (`Tensor`):
                Image features.
            latents (`Tensor`):
                Latent features.
            timestep_emb (`Tensor`):
                Timestep embedding.

        Returns:
            `Tensor`: Output latent features.
        """

        # Shift and scale for AdaLayerNorm
        emb = self.adaln_proj(self.adaln_silu(timestep_emb))
        shift_msa, scale_msa, shift_mlp, scale_mlp = emb.chunk(4, dim=1)

        # Fused Attention
        residual = latents
        x = self.ln0(x)
        latents = self.ln1(latents) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        batch_size = latents.shape[0]

        query = self.attn.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        key, value = self.attn.to_kv(kv_input).chunk(2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        weight = (query * self.attn.scale) @ (key * self.attn.scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        latents = weight @ value

        latents = latents.transpose(1, 2).reshape(
            batch_size, -1, self.attn.heads * head_dim
        )
        latents = self.attn.to_out[0](latents)
        latents = self.attn.to_out[1](latents)
        latents = latents + residual

        ## FeedForward
        residual = latents
        latents = (
            self.adaln_norm(latents) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        return self.ff(latents) + residual


# Modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
class IPAdapterTimeImageProjection(nn.Module):
    """Resampler of SD3 IP-Adapter with timestep embedding.

    Args:
        embed_dim (`int`, defaults to 1152):
            The feature dimension.
        output_dim (`int`, defaults to 2432):
            The number of output channels.
        hidden_dim (`int`, defaults to 1280):
            The number of hidden channels.
        depth (`int`, defaults to 4):
            The number of blocks.
        dim_head (`int`, defaults to 64):
            The number of head channels.
        heads (`int`, defaults to 20):
            Parallel attention heads.
        num_queries (`int`, defaults to 64):
            The number of queries.
        ffn_ratio (`int`, defaults to 4):
            The expansion ratio of feedforward network hidden layer channels.
        timestep_in_dim (`int`, defaults to 320):
            The number of input channels for timestep embedding.
        timestep_flip_sin_to_cos (`bool`, defaults to True):
            Flip the timestep embedding order to `cos, sin` (if True) or `sin, cos` (if False).
        timestep_freq_shift (`int`, defaults to 0):
            Controls the timestep delta between frequencies between dimensions.
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        output_dim: int = 2432,
        hidden_dim: int = 1280,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 20,
        num_queries: int = 64,
        ffn_ratio: int = 4,
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter([1, num_queries, hidden_dim], dtype=dtype)
        self.proj_in = nn.Linear(embed_dim, hidden_dim, dtype=dtype)
        self.proj_out = nn.Linear(hidden_dim, output_dim, dtype=dtype)
        self.norm_out = nn.LayerNorm(output_dim, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                IPAdapterTimeImageProjectionBlock(
                    hidden_dim, dim_head, heads, ffn_ratio, dtype=dtype
                )
                for _ in range(depth)
            ]
        )
        self.time_proj = Timesteps(
            timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift, dtype=dtype
        )
        self.time_embedding = TimestepEmbedding(
            timestep_in_dim, hidden_dim, act_fn="silu", dtype=dtype
        )

    def forward(self, x: Tensor, timestep: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x (`Tensor`):
                Image features.
            timestep (`Tensor`):
                Timestep in denoising process.
        Returns:
            `Tuple`[`Tensor`, `Tensor`]: The pair (latents, timestep_emb).
        """
        timestep_emb = ops.cast()(self.time_proj(timestep), dtype=x.dtype())
        timestep_emb: Tensor = self.time_embedding(timestep_emb)

        latents = self.latents.tensor()

        x = self.proj_in(x)
        x = x + timestep_emb[:, None]

        for block in self.layers:
            latents: Tensor = block(x, latents, timestep_emb)

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        return latents, timestep_emb


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
