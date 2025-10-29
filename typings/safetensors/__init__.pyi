from typing import Any, overload, Literal, Self, SupportsIndex, Sequence
from types import TracebackType

from numpy import ndarray
from torch import Tensor

FrameworkLiteral = Literal["pt", "tf", "flax", "numpy"]

class PySafeSlice[FW: FrameworkLiteral]:

    @overload
    def __getitem__(
        self: "PySafeSlice[Literal['pt']]",
        key: ndarray | Tensor | SupportsIndex | slice | Sequence[SupportsIndex]
        ) -> Tensor: ...
    @overload
    def __getitem__(
        self: "PySafeSlice[Literal['numpy']]",
        key: ndarray | SupportsIndex | slice | Sequence[SupportsIndex]
        ) -> ndarray: ...
    @overload
    def __getitem__(
        self: "PySafeSlice[Literal['tf', 'flax']]",
        key: Any
        ) -> Any: ...

class safe_open[FW: FrameworkLiteral]:

    def __init__(
        self,
        filename: str,
        framework: FW,
        device: str = "cpu",
        ) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
        ) -> None: ...

    def get_slice(self, name: str) -> PySafeSlice[FW]: ...

    @overload
    def get_tensor(self: "safe_open[Literal['pt']]", name: str) -> Tensor: ...
    @overload
    def get_tensor(self: "safe_open[Literal['numpy']]", name: str) -> ndarray: ...
    @overload
    def get_tensor(self: "safe_open[Literal['tf']]", name: str) -> Any: ...

    def keys(self) -> list[str]: ...

    def metadata(self) -> dict[str, str]: ...

    def offset_keys(self) -> list[str]: ...