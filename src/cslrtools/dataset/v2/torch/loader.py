# pyright: reportUnnecessaryIsInstance=false

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING, cast, Any,
    Iterator, Mapping, Hashable, Callable,
    TypeVar, Generic, Literal
)
from typing_extensions import TypeIs
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.serialization import MAP_LOCATION

_T = TypeVar("_T", bound=Hashable)
_A = TypeVar("_A")

def _is_array_map(
    obj: object, type_: type[Mapping[str | _T, _A]]
    ) -> TypeIs[Mapping[str | _T, _A]]:

    return isinstance(obj, type_)

class ArrayLoader(ABC, Generic[_T, _A]):
    """
    Abstract base class for generic array loaders.
    Loads tensors or mappings from data files,
    provides caching and type conversion (as_tensor).
    Subclasses implement the load method to define
    file format-specific loading logic.
    get_tensor enables key-based loading,
    enable_cache accelerates repeated access.
    Supports flexible handling of data types and keys.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        as_tensor: Callable[[_A], Tensor] = torch.tensor,
        ):
        self.cache: dict[Path, Mapping[str | _T, _A]] = {}
        self.default_key = default_key
        self.enable_cache = enable_cache

        @wraps(as_tensor)
        def as_tensor_wrapper(data: _A) -> Tensor:
            if not isinstance(data, Tensor):
                data = as_tensor(data)
            return data

        self.as_tensor = as_tensor_wrapper

    def clear_cache(self):
        self.cache.clear()

    @abstractmethod
    def load(self, path: Path) -> _A | Mapping[str | _T, _A]:
        pass

    def get_tensor(self, path: Path, key: str | _T | None = None) -> Tensor:

        if key is None:
            key = self.default_key

        if path in self.cache:
            return self.as_tensor(self.cache[path][key])

        data: _A | Mapping[str | _T, _A] = self.load(path)

        mapping: Mapping[str | _T, _A]
        if _is_array_map(data, Mapping[str | _T, _A]):
            mapping = data
        else:
            mapping = {self.default_key: data}

        if self.enable_cache:
            self.cache[path] = mapping

        return self.as_tensor(mapping[key])

class CsvLoader(ArrayLoader[_T, Tensor]):
    """
    Loader for CSV files as tensors.
    Allows specification of delimiter and reshape shape.
    Converts single arrays to torch.Tensor.
    Supports caching and key-based access.
    Suitable for lightweight numeric data loading.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        delimiter: str = ",",
        shape: tuple[int, ...] = (-1,)
        ):
        super().__init__(default_key, enable_cache)
        self.delimiter = delimiter
        self.shape = shape

    def load(self, path: Path) -> Tensor:

        data = np.loadtxt(path, delimiter=self.delimiter)
        return torch.tensor(data).reshape(self.shape)

class NpyLoader(ArrayLoader[_T, Tensor]):
    """
    Loader for NumPy npy files as tensors.
    No type conversion needed (default Tensor).
    Optimized for fast loading of single arrays.
    Supports caching and key-based access.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        ):
        super().__init__(
            default_key=default_key,
            enable_cache=enable_cache,
            as_tensor=lambda x: x,
        )

    def load(self, path: Path) -> Tensor:

        data = np.load(path)
        return torch.tensor(data)

class NpzLoader(ArrayLoader[_T, np.ndarray]):
    """
    Loader for NumPy npz (multiple arrays) files.
    Supports mapping type for multiple keys.
    Each array can be converted to torch.Tensor.
    Ensures type safety with _is_npz_file.
    Supports caching and key-based access.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        ):
        super().__init__(
            default_key,
            enable_cache=enable_cache,
            as_tensor=torch.tensor
        )

    def _is_npz_file(self, obj: object) -> TypeIs[np.lib.npyio.NpzFile[Any]]:
        return isinstance(obj, np.lib.npyio.NpzFile)

    def load(self, path: Path):

        data = np.load(path)
        assert self._is_npz_file(data)
        return cast(Mapping[str | _T, np.ndarray], data)

class PthLoader(ArrayLoader[_T, Tensor]):
    """
    Loader for PyTorch pth files (models/weights/tensors).
    Supports fine-grained control via map_location, pickle_mode, etc.
    Options like weights_only and mmap are also supported.
    Handles both single tensors and mapping types.
    Supports caching and key-based access.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        map_location: MAP_LOCATION = None,
        pickle_mode: Any = None,
        *,
        weights_only: bool | None = None,
        mmap: bool | None = None,
        **pickle_load_args: Any
        ):
        super().__init__(
            default_key,
            enable_cache=enable_cache,
            as_tensor=lambda x: x
        )
        self.map_location = map_location
        self.pickle_mode = pickle_mode
        self.weights_only = weights_only
        self.mmap = mmap
        self.pickle_load_args = pickle_load_args

    def load(self, path: Path) -> Tensor | Mapping[str | _T, Tensor]:

        data = torch.load(
            path,
            map_location=self.map_location,
            pickle_mode=self.pickle_mode,
            weights_only=self.weights_only,
            mmap=self.mmap,
            **self.pickle_load_args,
        )

        if isinstance(data, Tensor):
            return data

        if _is_array_map(data, Mapping[str | _T, Tensor]):
            return data

        raise TypeError(f"Unsupported data type: {type(data)}")

if TYPE_CHECKING:
    from safetensors import safe_open
else:
    safe_open = Any

class SafetensorMap(Mapping[str | _T, Tensor]):
    """
    Mapping wrapper for safetensors files.
    Retrieves tensors by key.
    Supports default_key for default selection.
    Abstracts safetensors API.
    Provides type-safe access and iteration.
    """
    def __init__(
        self,
        default_key: _T | None,
        safetensors: "safe_open[Literal['pt']]"
        ):
        self.default_key = default_key
        self.safetensors = safetensors

    def __getitem__(self, key: str | _T, /) -> Tensor:


        if isinstance(key, str):
            pass
        elif isinstance(key, type(self.default_key)):
            raise KeyError(f"Key {key} is type of default_key, but safetensors keys are str.")
        else:
            raise KeyError(f"key {key} is invalid type.")

        return self.safetensors.get_tensor(key)

    def __iter__(self) -> Iterator[str | _T]:
        return iter(self.safetensors.keys())

    def __len__(self) -> int:
        return len(self.safetensors.keys())

class SafetensorLoader(ArrayLoader[_T, Tensor]):
    """
    Loader for safetensors format files.
    Uses safe_open API for loading.
    Supports retrieval of multiple tensors as a mapping.
    Supports caching and key-based access.
    Raises ImportError if safetensors is not installed.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        ):

        super().__init__(
            default_key,
            enable_cache=enable_cache,
            as_tensor=lambda x: x
        )

        try:
            from safetensors import safe_open
            self.safe_open = safe_open
        except ImportError:
            raise ImportError(
                "safetensors is not installed."
                " Please install safetensors to use SafetensorLoader."
            )

    def load(self, path: Path):

        safetensor_file = self.safe_open(
            str(path),
            framework="pt",
        )

        return SafetensorMap(
            default_key=self.default_key,
            safetensors=safetensor_file
        )

####### video loader #######

class VideoLoader(ArrayLoader[_T, Tensor]):
    """
    Loader for video files (e.g., mp4, avi) as tensors.
    Flexible preprocessing and type conversion via as_tensor.
    Handles both single tensors and mapping types.
    Supports caching and key-based access.
    Suitable for batch processing of video data.
    """

    def __init__(
        self,
        default_key: _T = None,
        enable_cache: bool = False,
        as_tensor: Callable[[Tensor], Tensor] = lambda x: x,
        ):
        super().__init__(
            default_key,
            enable_cache=enable_cache,
            as_tensor=as_tensor
        )

    def load(self, path: Path) -> Tensor | Mapping[str | _T, Tensor]:

        import torchvision.io

        video, _, _ = torchvision.io.read_video(
            str(path),
            pts_unit="sec",
            output_format="TCHW",
        )

        return video.unsqueeze(0)  # (1, T, C, H, W)