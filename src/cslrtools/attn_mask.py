import torch
from torch import Tensor

# for copilot
# GoogleスタイルのdocstringをpublicAPIに付与
# すべての項目で (改行) プレースホルダー: (`型`)とし、型は必ず``で囲む
# Args, Returns, Raises(あれば), Exampleの順に記載
# 型が `Tensor` の場合は ":" の直後に形状を記載
# 例外をスローする前の日本語のコメントをもとに英語のメッセージを作成

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
