from typing import Optional, Tuple, Union

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..modeling_outputs import AutoencoderKLOutput

from .vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
    Encoder,
    MaskConditionDecoder,
)


class AsymmetricAutoencoderKL(nn.Module):
    r"""
    Designing a Better Asymmetric VQGAN for StableDiffusion https://arxiv.org/abs/2306.04632 . A VAE model with KL loss
    for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        down_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of down block output channels.
        layers_per_down_block (`int`, *optional*, defaults to `1`):
            Number layers for down block.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        up_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of up block output channels.
        layers_per_up_block (`int`, *optional*, defaults to `1`):
            Number layers for up block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        norm_num_groups (`int`, *optional*, defaults to `32`):
            Number of groups to use for the first normalization layer in ResNet blocks.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        down_block_out_channels: Tuple[int, ...] = (64,),
        layers_per_down_block: int = 1,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        up_block_out_channels: Tuple[int, ...] = (64,),
        layers_per_up_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        dtype: str = "float16",
        **kwargs,
    ) -> None:
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=down_block_out_channels,
            layers_per_block=layers_per_down_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            dtype=dtype,
        )

        # pass init params to Decoder
        self.decoder = MaskConditionDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=up_block_out_channels,
            layers_per_block=layers_per_up_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            dtype=dtype,
        )

        self.quant_conv = nn.Conv2d(
            2 * latent_channels, 2 * latent_channels, 1, dtype=dtype
        )
        self.post_quant_conv = nn.Conv2d(
            latent_channels,
            latent_channels,
            1,
            dtype=dtype,
        )

        self.use_slicing = False
        self.use_tiling = False

    def encode(
        self, x: Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[Tensor]]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: Tensor,
        image: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[Tensor]]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, image, mask)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self,
        z: Tensor,
        generator=None,
        image: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[Tensor]]:
        decoded = self._decode(z, image, mask).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: Tensor,
        mask: Optional[Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator=None,
    ) -> Union[DecoderOutput, Tuple[Tensor]]:
        r"""
        Args:
            sample (`Tensor`): Input sample.
            mask (`Tensor`, *optional*, defaults to `None`): Optional inpainting mask.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, generator, sample, mask).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
