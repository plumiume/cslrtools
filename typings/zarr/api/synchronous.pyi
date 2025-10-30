from typing import TypeAlias, Any, Literal, Iterable
from typing_extensions import deprecated
from pathlib import Path
from fsspec.mapping import FSMap as _FSMap

import numpy as np
import numpy.typing as npt

from zarr import Group
from zarr.api.asynchronous import ArrayLike
from zarr.storage._common import Store, StorePath, Buffer
from zarr.abc.numcodec import Numcodec
from zarr.abc.codec import Codec
from zarr.core.common import (
    AccessModeLiteral, ZarrFormat, JSON, Any, MemoryOrder, DimensionNames,
    ShapeLike
)
from zarr.core.array import (
    Array, NDArrayLike, CompressorLike, DEFAULT_FILL_VALUE, ShardsLike,
    FiltersLike, CompressorsLike, SerializerLike, ChunkKeyEncodingLike
)
from zarr.core.array_spec import ArrayConfigLike
from zarr.core.buffer.core import NDArrayLikeOrScalar
from zarr.core.chunk_key_encodings import ChunkKeyEncoding
from zarr.core.dtype import ZDTypeLike
from zarr.core.sync_group import create_hierarchy
from zarr.errors import ZarrDeprecationWarning


__all__ = [
    "array",
    "consolidate_metadata",
    "copy",
    "copy_all",
    "copy_store",
    "create",
    "create_array",
    "create_hierarchy",
    "empty",
    "empty_like",
    "from_array",
    "full",
    "full_like",
    "group",
    "load",
    "ones",
    "ones_like",
    "open",
    "open_array",
    "open_consolidated",
    "open_group",
    "open_like",
    "save",
    "save_array",
    "save_group",
    "tree",
    "zeros",
    "zeros_like",
]



FSMap: TypeAlias = _FSMap | None
StoreLike: TypeAlias = Store | StorePath | FSMap | Path | str | dict[str, Buffer]


def consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    ) -> Group: ...


def copy(*args: Any, **kwargs: Any) -> tuple[int, int, int]: ...


def copy_all(*args: Any, **kwargs: Any) -> tuple[int, int, int]: ...


def copy_store(*args: Any, **kwargs: Any) -> tuple[int, int, int]: ...


def load(
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    zarr_version: ZarrFormat | None = None,
    ) -> NDArrayLikeOrScalar | dict[str, NDArrayLikeOrScalar]: ...


def open(
    store: StoreLike | None = None,
    *,
    mode: AccessModeLiteral | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
    ) -> Array | Group: ...


def open_consolidated(*args: Any, use_consolidated: Literal[True] = True, **kwargs: Any) -> Group: ...


def save(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,
    ) -> None: ...


def save_array(
    store: StoreLike,
    arr: NDArrayLike,
    *,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
    ) -> None: ...


def save_group(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: NDArrayLike,
    ) -> None: ...


@deprecated("Use Group.tree instead.", category=ZarrDeprecationWarning)
def tree(grp: Group, expand: bool | None = None, level: int | None = None) -> Any: ...


def array(data: npt.ArrayLike | Array, **kwargs: Any) -> Array: ...


def group(
    store: StoreLike | None = None,
    *,
    overwrite: bool = False,
    chunk_store: StoreLike | None = None,  # not used
    cache_attrs: bool | None = None,  # not used, default changed
    synchronizer: Any | None = None,  # not used
    path: str | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
    attributes: dict[str, JSON] | None = None,
    storage_options: dict[str, Any] | None = None,
    ) -> Group: ...


def open_group(
    store: StoreLike | None = None,
    *,
    mode: AccessModeLiteral = "a",
    cache_attrs: bool | None = None,
    synchronizer: Any = None,
    path: str | None = None,
    chunk_store: StoreLike | None = None,
    storage_options: dict[str, Any] | None = None,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,
    attributes: dict[str, JSON] | None = None,
    use_consolidated: bool | str | None = None,
    ) -> Group: ...


def create_group(
    store: StoreLike,
    *,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    overwrite: bool = False,
    attributes: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
    ) -> Group: ...


def create(
    shape: tuple[int, ...] | int,
    *,  # Note: this is a change from v2
    chunks: tuple[int, ...] | int | bool | None = None,
    dtype: ZDTypeLike | None = None,
    compressor: CompressorLike = "auto",
    fill_value: Any | None = DEFAULT_FILL_VALUE,  # TODO: need type
    order: MemoryOrder | None = None,
    store: StoreLike | None = None,
    synchronizer: Any | None = None,
    overwrite: bool = False,
    path: str | None = None,
    chunk_store: StoreLike | None = None,
    filters: Iterable[dict[str, JSON] | Numcodec] | None = None,
    cache_metadata: bool | None = None,
    cache_attrs: bool | None = None,
    read_only: bool | None = None,
    object_codec: Codec | None = None,  # TODO: type has changed
    dimension_separator: Literal[".", "/"] | None = None,
    write_empty_chunks: bool | None = None,  # TODO: default has changed
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # TODO: need type
    attributes: dict[str, JSON] | None = None,
    # v3 only
    chunk_shape: tuple[int, ...] | int | None = None,
    chunk_key_encoding: (
        ChunkKeyEncoding
        | tuple[Literal["default"], Literal[".", "/"]]
        | tuple[Literal["v2"], Literal[".", "/"]]
        | None
    ) = None,
    codecs: Iterable[Codec | dict[str, JSON]] | None = None,
    dimension_names: DimensionNames = None,
    storage_options: dict[str, Any] | None = None,
    config: ArrayConfigLike | None = None,
    **kwargs: Any,
    ) -> Array: ...


def create_array(
    store: StoreLike,
    *,
    name: str | None = None,
    shape: ShapeLike | None = None,
    dtype: ZDTypeLike | None = None,
    data: np.ndarray[Any, np.dtype[Any]] | None = None,
    chunks: tuple[int, ...] | Literal["auto"] = "auto",
    shards: ShardsLike | None = None,
    filters: FiltersLike = "auto",
    compressors: CompressorsLike = "auto",
    serializer: SerializerLike = "auto",
    fill_value: Any | None = DEFAULT_FILL_VALUE,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat | None = 3,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncodingLike | None = None,
    dimension_names: DimensionNames = None,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    config: ArrayConfigLike | None = None,
    write_data: bool = True,
    ) -> Array: ...


def from_array(
    store: StoreLike,
    *,
    data: Array | npt.ArrayLike,
    write_data: bool = True,
    name: str | None = None,
    chunks: Literal["auto", "keep"] | tuple[int, ...] = "keep",
    shards: ShardsLike | None | Literal["keep"] = "keep",
    filters: FiltersLike | Literal["keep"] = "keep",
    compressors: CompressorsLike | Literal["keep"] = "keep",
    serializer: SerializerLike | Literal["keep"] = "keep",
    fill_value: Any | None = DEFAULT_FILL_VALUE,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat | None = None,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncodingLike | None = None,
    dimension_names: DimensionNames = None,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    config: ArrayConfigLike | None = None,
    ) -> Array: ...


def empty(shape: tuple[int, ...], **kwargs: Any) -> Array: ...


def empty_like(a: ArrayLike, **kwargs: Any) -> Array: ...


def full(shape: tuple[int, ...], fill_value: Any, **kwargs: Any) -> Array: ...


def full_like(a: ArrayLike, **kwargs: Any) -> Array: ...


def ones(shape: tuple[int, ...], **kwargs: Any) -> Array: ...


def ones_like(a: ArrayLike, **kwargs: Any) -> Array: ...


def open_array(
    store: StoreLike | None = None,
    *,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str = "",
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
    ) -> Array: ...


def open_like(a: ArrayLike, path: str, **kwargs: Any) -> Array: ...


def zeros(shape: tuple[int, ...], **kwargs: Any) -> Array: ...


def zeros_like(a: ArrayLike, **kwargs: Any) -> Array: ...





