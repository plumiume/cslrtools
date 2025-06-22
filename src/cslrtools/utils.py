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

from torch import Size
from functools import reduce
import operator

def batch_check(*batches: Size) -> Size:
    """
    Check if all provided batch sizes are the same.

    Args:
        *batches (`torch.Size`): Variable number of batch sizes to check.

    Returns:
        out (`torch.Size`): The common batch size if all are the same.

    Raises:
        raise (`ValueError`): If the batch sizes are not the same.
    """

    if not batches:
        return Size()

    b0, *bs = batches

    for i, b in enumerate(bs):
        if b != b0:
            raise ValueError(f"Batch size mismatch: {b0} at index 0 != {b} at index {i + 1}")

    return b0

def batch_product(batch: Size) -> int:

    return reduce(operator.mul, batch, 1)
