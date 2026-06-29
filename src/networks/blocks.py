from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, conv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = conv

    def forward(self, x):
        return self.conv(x)

class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
    ):
        super().__init__()
        self.conv1 = Convolution(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False))
        self.conv2 = Convolution(nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=1, bias=False))
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = Convolution(nn.Conv3d(in_channels, out_channels, 1, stride, bias=False))
            self.norm3 = nn.InstanceNorm3d(num_features=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out
    

class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = True,
    ) -> None:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            pass

    def forward(self, inp):
        return self.layer(inp)


class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = True,
    ) -> None:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = Convolution(nn.ConvTranspose3d(in_channels, out_channels, upsample_kernel_size, upsample_stride, bias=False))

        if res_block:
            self.conv_block = UnetResBlock(
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            pass

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
    

class UnetOutBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, dropout: tuple | str | float | None = None
    ):
        super().__init__()
        self.conv = Convolution(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=True))

    def forward(self, inp):
        return self.conv(inp)
