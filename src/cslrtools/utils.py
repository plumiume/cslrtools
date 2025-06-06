from torch import Size
from functools import reduce
import operator

# for copilot
# GoogleスタイルのdocstringをpublicAPIに付与
# 例外をスローする前の日本語のコメントをもとに英語のメッセージを作成

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
