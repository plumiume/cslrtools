from os import PathLike
from typing import AnyStr
from torch import Tensor

def load_file(
    filename: str | PathLike[AnyStr],
    device: str | int = "cpu",
    ) -> dict[str, Tensor]: ...
