from typing import (
    TypeVar, Generic, overload,
    Sequence
)
import torch
import torch.nn as nn
from torch import Tensor, Size

# for copilot
# GoogleスタイルのdocstringをpublicAPIに付与
# すべての項目で (改行) プレースホルダー: (`型`)とし、型は必ず``で囲む
# Args, Returns, Raises(あれば), Exampleの順に記載
# 型が `Tensor` の場合は ":" の直後に形状を記載
# 例外をスローする前の日本語のコメントをもとに英語のメッセージを作成

def conv_size(
    size: Tensor,
    kernel_size: Tensor,
    stride: Tensor,
    padding: Tensor,
    dilation: Tensor
    ) -> Tensor:

    return torch.floor_divide(
        size + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride
    ) + 1

def conv_transpose_size(
    size: Tensor,
    kernel_size: Tensor,
    stride: Tensor,
    padding: Tensor,
    output_padding: Tensor,
    dilation: Tensor
    ) -> Tensor:

    return (
        (size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding + 1
    )

_ConvNd = TypeVar('_ConvNd', bound=nn.Conv1d | nn.Conv2d | nn.Conv3d)
_ConvTransposeNd = TypeVar('_ConvTransposeNd', bound=nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d)

class ConvSize(nn.Module, Generic[_ConvNd]):

    """
    Calculate the output size of a convolutional layer.

    Parameters:
        conv (`_ConvNd`): A convolutional layer instance (e.g., `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`).

    Shape:
        - Input: `Tensor` of shape `(batch_size, channels, ...)` or `Size`.
        - Output: `Tensor` of shape `(batch_size, out_channels, ...)` or `Size`.
    """

    def __init__(self, conv: _ConvNd):
        super().__init__()
        self.kernel_size = nn.Parameter(torch.tensor(conv.kernel_size), requires_grad=False)
        self.stride = nn.Parameter(torch.tensor(conv.stride), requires_grad=False)
        self.padding = nn.Parameter(torch.tensor(conv.padding), requires_grad=False)
        self.dilation = nn.Parameter(torch.tensor(conv.dilation), requires_grad=False)

    @overload
    def forward(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor: ...

    @overload
    def forward(
        self,
        shape: Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Size: ...

    def forward( # type: ignore[reportInconsistentOverload]
        self,
        arg: Tensor | Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor | Size:

        if isinstance(arg, Tensor):
            return self._forward_tensor(arg, dim)
        elif isinstance(arg, Size):
            return self._forward_size(arg)
        else:
            raise TypeError(f"Expected Tensor or Size, got {type(arg).__name__}")

    def _forward_tensor(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor:
        return conv_size(
            size,
            self.kernel_size[dim],
            self.stride[dim],
            self.padding[dim],
            self.dilation[dim]
        )

    def _forward_size(
        self,
        shape: Size
        ) -> Size:

        change_ndim = len(self.kernel_size)
        change_shape = shape[-change_ndim:]
        unchange_shape = shape[:-change_ndim]
        changed_shape = Size(conv_size(
            torch.tensor(change_shape),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        ).tolist())
        return Size(unchange_shape + changed_shape)

class ConvTransposeSize(nn.Module, Generic[_ConvTransposeNd]):

    """
    Calculate the output size of a transposed convolutional layer.

    Parameters:
        conv (`_ConvTransposeNd`): A transposed convolutional layer instance (e.g., `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, `nn.ConvTranspose3d`).

    Shape:
        - Input: `Tensor` of shape `(batch_size, channels, ...)` or `Size`.
        - Output: `Tensor` of shape `(batch_size, out_channels, ...)` or `Size`.
    """

    def __init__(self, conv: _ConvTransposeNd):
        super().__init__()
        self.kernel_size = nn.Parameter(torch.tensor(conv.kernel_size), requires_grad=False)
        self.stride = nn.Parameter(torch.tensor(conv.stride), requires_grad=False)
        self.padding = nn.Parameter(torch.tensor(conv.padding), requires_grad=False)
        self.output_padding = nn.Parameter(torch.tensor(conv.output_padding), requires_grad=False)
        self.dilation = nn.Parameter(torch.tensor(conv.dilation), requires_grad=False)

    @overload
    def forward(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor: ...

    @overload
    def forward(
        self,
        shape: Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Size: ...

    def forward( # type: ignore[reportInconsistentOverload]
        self,
        arg: Tensor | Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor | Size:

        if isinstance(arg, Tensor):
            return self._forward_tensor(arg, dim)
        elif isinstance(arg, Size):
            return self._forward_size(arg)
        else:
            raise TypeError(f"Expected Tensor or Size, got {type(arg).__name__}")

    def _forward_tensor(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor:
        return conv_transpose_size(
            size,
            self.kernel_size[dim],
            self.stride[dim],
            self.padding[dim],
            self.output_padding[dim],
            self.dilation[dim]
        )

    def _forward_size(
        self,
        shape: Size
        ) -> Size:

        change_ndim = len(self.kernel_size)
        change_shape = shape[-change_ndim:]
        unchange_shape = shape[:-change_ndim]
        changed_shape = Size(conv_transpose_size(
            torch.tensor(change_shape),
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation
        ).tolist())
        return Size(unchange_shape + changed_shape)
