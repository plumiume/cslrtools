# Copyright 2025 plumiume.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor
import torch
import torchaudio

from .utils import batch_check, batch_product

def edit_distance(
    a: Tensor,
    alen: Tensor,
    b: Tensor,
    blen: Tensor,
    ) -> Tensor:
    """
    Calculate the edit distance between two sequences.

    Args:
        a (`Tensor`): First sequence of shape (..., alen).
        alen (`Tensor`): Lengths of the first sequence of shape (...).
        b (`Tensor`): Second sequence of shape (..., blen).
        blen (`Tensor`): Lengths of the second sequence of shape (...).

    Returns:
        out (`Tensor`): Edit distance of shape (...).

    Example:
        >>> a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> alen = torch.tensor([3, 3])
        >>> b = torch.tensor([[1, 2], [4, 5, 6]])
        >>> blen = torch.tensor([2, 3])
        >>> edit_distance(a, alen, b, blen)
        tensor([1, 0])
    """

    batch = batch_check(a.shape[:-1], alen.shape, b.shape[:-1], blen.shape)
    size = batch_product(batch)

    a_reshaped = a.reshape(size, -1)
    alen_reshaped = alen.reshape(size)
    b_reshaped = b.reshape(size, -1)
    blen_reshaped = blen.reshape(size)

    o = torch.tensor([
        torchaudio.functional.edit_distance(
            ai[:aleni].tolist(), bi[:bleni].tolist()
        )
        for ai, aleni, bi, bleni in zip(
            a_reshaped, alen_reshaped, b_reshaped, blen_reshaped
        )
    ], device=a.device)

    return o.reshape(batch)

def wer(
    h: Tensor,
    hlen: Tensor,
    r: Tensor,
    rlen: Tensor,
    normalize: bool = True,
    ) -> Tensor:

    """
    Calculate the Word Error Rate (WER) between two sequences.

    Args:
        h (`Tensor`): Hypothesis sequence of shape (..., hlen).
        hlen (`Tensor`): Lengths of the hypothesis sequence of shape (...).
        r (`Tensor`): Reference sequence of shape (..., rlen).
        rlen (`Tensor`): Lengths of the reference sequence of shape (...).
        normalize (`bool`, optional): If True, normalizes the WER by the length of the reference. Defaults to True.

    Returns:
        out (`Tensor`): Word Error Rate of shape (...).

    Example:
        >>> h = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> hlen = torch.tensor([3, 3])
        >>> r = torch.tensor([[1, 2], [4, 5, 6]])
        >>> rlen = torch.tensor([2, 3])
        >>> wer(h, hlen, r, rlen)
        tensor([0.5, 0.0])
    """

    edist = edit_distance(h, hlen, r, rlen)
    if normalize:
        return edist / rlen.float()
    else:
        return edist
    