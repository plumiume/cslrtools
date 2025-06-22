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

import torch
from torch import Tensor

def attn_mask(
    lengths: Tensor,
    max_len: int | None = None,
    ):

    """
    Create an attention mask based on the given lengths.

    Args:
        lengths (`Tensor`): A tensor of shape (..., )
            representing the lengths of sequences.
        max_len (`int`, optional): The maximum length to consider for the mask.
            If `None`, it will be set to the maximum value in `lengths`.

    Returns:
        out (`Tensor`): A boolean tensor of shape (..., max_len)
            where `True` indicates positions that are valid
            (up to the lengths specified)
            and `False` indicates positions that are invalid
            (beyond the lengths).
    """

    if max_len is None:
        max_len = int(lengths.max().item())

    arange = torch.arange(max_len)
    return lengths.unsqueeze(-1) <= arange
