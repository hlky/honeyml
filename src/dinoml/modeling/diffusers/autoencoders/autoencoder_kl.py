from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from ....compiler import ops
from ....frontend import nn, Tensor
from ....utils.build_utils import Shape, DimAdd, DimDiv, DimMul, DimSub

from ..attention_processor import AttentionProcessor
from ..modeling_outputs import AutoencoderKLOutput

from .vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder


class AutoencoderKL(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            dtype=dtype,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            dtype=dtype,
        )

        self.quant_conv = (
            nn.Conv2d(
                2 * latent_channels,
                2 * latent_channels,
                1,
                dtype=dtype,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(
                latent_channels,
                latent_channels,
                1,
                dtype=dtype,
            )
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def encode(
        self,
        x: Annotated[
            Tensor,
            (
                Shape(name="batch_size"),
                Shape(name="height"),
                Shape(name="width"),
                Shape(name="channels", config_name="out_channels"),
            ),
        ],
        sample: Annotated[
            Tensor,
            (
                Shape(name="batch_size"),
                Shape(name="height", dim_operations=(DimDiv(8),)),
                Shape(name="width", dim_operations=(DimDiv(8),)),
                Shape(name="channels", config_name="latent_channels"),
            ),
        ],
        sample_mode: Union[
            Literal["sample", "argmax", "moments"],
            List[Literal["sample", "argmax", "moments"]],
        ] = "sample",
        return_dict: bool = True,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_tiling and (
            ops.size()(x, dim=1) > self.tile_sample_min_size
            or ops.size()(x, dim=2) > self.tile_sample_min_size
        ):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and ops.size()(x, dim=0) > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in ops.split()(x, 1)]
            h = ops.concatenate()(encoded_slices)
        else:
            h = self.encoder(x)

        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h

        posterior = DiagonalGaussianDistribution(moments)

        if not isinstance(sample_mode, list):
            sample_mode = [sample_mode]

        outputs: List[Tensor] = []

        for sample_type in sample_mode:
            if sample_type == "moments":
                outputs.append(moments)
            elif sample_type == "argmax":
                outputs.append(posterior.mode())
            elif sample_type == "sample":
                outputs.append(posterior.sample(sample=sample))

        if not return_dict:
            return tuple(outputs)

        if len(outputs) == 1:
            outputs: Tensor = outputs[0]

        return outputs

    def _decode(
        self,
        z: Annotated[
            Tensor,
            (
                Shape(name="batch_size", step=1),
                Shape(name="height", dim_operations=(DimDiv(8),), step=8),
                Shape(name="width", dim_operations=(DimDiv(8),), step=8),
                Shape(name="channels", config_name="latent_channels"),
            ),
        ],
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tensor]:
        if self.use_tiling and (
            ops.size()(z, dim=1) > self.tile_sample_min_size
            or ops.size()(z, dim=2) > self.tile_sample_min_size
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self,
        z: Annotated[
            Tensor,
            (
                Shape(name="batch_size"),
                Shape(name="height", dim_operations=(DimDiv(8),)),
                Shape(name="width", dim_operations=(DimDiv(8),)),
                Shape(name="channels", config_name="latent_channels"),
            ),
        ],
        return_dict: bool = True,
        generator=None,
    ) -> Union[DecoderOutput, Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if self.use_slicing and ops.size()(z, dim=0) > 1:
            decoded_slices = [
                self._decode(z_slice).sample for z_slice in ops.split()(z, 1)
            ]
            decoded = ops.concatenate()(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, y, :
            ] * (y / blend_extent)
        return b

    def blend_h(self, a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    def tiled_encode(self, x: Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, ops.size()(x, dim=1), overlap_size):
            row = []
            for j in range(0, ops.size()(x, dim=2), overlap_size):
                tile = ops.dynamic_slice()(
                    x,
                    start_indices=[0, i, j, 0],
                    end_indices=[
                        None,
                        i + self.tile_sample_min_size,
                        j + self.tile_sample_min_size,
                        None,
                    ],
                )
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(ops.concatenate()(result_row, dim=2))

        moments = ops.concatenate()(result_rows, dim=1)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(
        self, z: Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(ops.concatenate()(result_row, dim=2))

        dec = ops.concatenate()(result_rows, dim=1)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator=None,
    ) -> Union[DecoderOutput, Tensor]:
        r"""
        Args:
            sample (`Tensor`): Input sample.
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
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
