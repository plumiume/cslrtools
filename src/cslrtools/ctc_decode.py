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

from typing import cast, overload, Literal, Iterable, Iterator
import torch
from torch import Tensor

from .utils import batch_check, batch_product

def _inverse(inverse_iter: Iterable[Tensor]) -> Tensor:
    return torch.stack(list(inverse_iter))

def _counts(counts_iter: Iterable[Tensor]) -> Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        list(counts_iter),
        batch_first=True,
        padding_value=0
    )


# return_inverse=False, return_counts=False
@overload
def ctc_decode( # type: ignore[reportOverlappingOverload]
    x_argmaxed: Tensor,
    xlen: Tensor,
    blank: int = 0,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    ) -> tuple[Tensor, Tensor]: ...

# return_inverse=True, return_counts=False
@overload
def ctc_decode(
    x_argmaxed: Tensor,
    xlen: Tensor,
    blank: int = 0,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[False] = False,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

# return_inverse=False, return_counts=True
@overload
def ctc_decode(
    x_argmaxed: Tensor,
    xlen: Tensor,
    blank: int = 0,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True] = ...,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

# return_inverse=True, return_counts=True
@overload
def ctc_decode(
    x_argmaxed: Tensor,
    xlen: Tensor,
    blank: int = 0,
    return_inverse: Literal[True] = ...,
    return_counts: Literal[True] = ...,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

def ctc_decode(
    x_argmaxed: Tensor,
    xlen: Tensor,
    blank: int = 0,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Decode CTC outputs by removing blanks and returning unique sequences.

    Args:
        x_argmaxed (`torch.Tensor`): The tensor of shape `(batch, time)` containing argmax indices.
        xlen (`torch.Tensor`): The tensor of shape `(batch,)` containing lengths of each sequence.
        blank (`int`, optional): The index representing the blank token. Defaults to 0.
        return_inverse (`bool`, optional): If True, returns the inverse indices of the unique sequences. Defaults to False.
        return_counts (`bool`, optional): If True, returns the counts of each unique sequence. Defaults to False.

    Returns:
        tuple: Depending on the flags `return_inverse` and `return_counts`, returns:
            - If both are False: `(unique_sequences, sequence_lengths)`
            - If `return_inverse` is True: `(unique_sequences, sequence_lengths, inverse_indices)`
            - If `return_counts` is True: `(unique_sequences, sequence_lengths, counts)`
            - If both are True: `(unique_sequences, sequence_lengths, inverse_indices, counts)`
    """

    batch = batch_check(x_argmaxed.shape[:-2], xlen.shape)
    size = batch_product(batch)

    x_reshaped = x_argmaxed.reshape(size, -1)
    xlen_reshaped = xlen.reshape(size)

    x_cutted = (xi[:xleni] for xi, xleni in zip(x_reshaped, xlen_reshaped))
    x_blank_removed = (xi[xi != blank] for xi in x_cutted)

    results = cast(Iterator[tuple[Tensor, Tensor, Tensor]], (
        torch.unique_consecutive( # pyright: ignore[reportUnknownMemberType]
            xi, return_inverse=True, return_counts=True
        )
        for xi in x_blank_removed
    ))

    o = torch.nn.utils.rnn.pad_sequence(
        [r[0] for r in results],
        batch_first=True,
        padding_value=blank
    ).reshape(*batch, -1)
    olen = (o != blank).sum(dim=-1)

    if return_inverse:
        if return_counts:
            inverse = _inverse(r[1] for r in results).reshape(*batch, -1)
            counts = _counts(r[2] for r in results).reshape(*batch, -1)
            return o, olen, inverse, counts
        else:
            inverse = _inverse(r[1] for r in results).reshape(*batch, -1)
            return o, olen, inverse
    else:
        if return_counts:
            counts = _counts(r[2] for r in results).reshape(*batch, -1)
            return o, olen, counts
        else:
            return o, olen
