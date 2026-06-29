import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from networks.transformers import SwinTransformer
from networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = (7, 7, 7),
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        downsample: str | nn.Module = "merging",
        use_v2: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            patch_size: size of the patch token.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            norm_layer: normalization layer.
            patch_norm: whether to apply normalization to the patch embedding. Default is False.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(in_channels=3, out_channels=2, use_checkpoint=True)

        """

        super().__init__()

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.patch_size = patch_size

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            downsample=downsample,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        layers1_0: BasicLayer = self.swinViT.layers1[0]  # type: ignore[assignment]
        layers2_0: BasicLayer = self.swinViT.layers2[0]  # type: ignore[assignment]
        layers3_0: BasicLayer = self.swinViT.layers3[0]  # type: ignore[assignment]
        layers4_0: BasicLayer = self.swinViT.layers4[0]  # type: ignore[assignment]
        wstate = weights["state_dict"]

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(wstate["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(wstate["module.patch_embed.proj.bias"])
            for bname, block in layers1_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")  # type: ignore[operator]

            if layers1_0.downsample is not None:
                d = layers1_0.downsample
                d.reduction.weight.copy_(wstate["module.layers1.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers1.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers1.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers2_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")  # type: ignore[operator]

            if layers2_0.downsample is not None:
                d = layers2_0.downsample
                d.reduction.weight.copy_(wstate["module.layers2.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers2.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers2.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers3_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")  # type: ignore[operator]

            if layers3_0.downsample is not None:
                d = layers3_0.downsample
                d.reduction.weight.copy_(wstate["module.layers3.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers3.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers3.0.downsample.norm.bias"])  # type: ignore

            for bname, block in layers4_0.blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")  # type: ignore[operator]

            if layers4_0.downsample is not None:
                d = layers4_0.downsample
                d.reduction.weight.copy_(wstate["module.layers4.0.downsample.reduction.weight"])  # type: ignore
                d.norm.weight.copy_(wstate["module.layers4.0.downsample.norm.weight"])  # type: ignore
                d.norm.bias.copy_(wstate["module.layers4.0.downsample.norm.bias"])  # type: ignore

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits
    