from abc import ABC, abstractmethod
from typing import (
    Any, TypeVar, ParamSpec, Generic,
    cast, Mapping, TypeAlias,
    Callable, Self, Iterator,
    
)

from functools import wraps
from contextlib import contextmanager
from os import PathLike as _PathLike
from pathlib import Path
from threading import local
from multiprocessing.context import BaseContext

import torch
from torch import Tensor
from torch.utils.data import (
    Dataset as _Dataset,
    IterableDataset as _IterableDataset,
    DataLoader
)

import zarr
from fsspec.mapping import FSMap as _FSMap
from zarr.storage._common import Store, StorePath, Buffer
from zarr.core.common import JSON


# ルートグループ(Group)
# metadata: メタデータグループ cslrtoolsシステム外の情報を格納、名前空間の衝突を避けるため
# {idx}: アイテムグループ

# アイテムグループ(Group)
# videos.{kvid}: Array
# landmarks.{klm}.landmark: Array
# landmarks.{klm}.connection: Array
# connections.{klm1}.{klm2}: Array
# targets.{ktgt}: Array

PathLike = _PathLike[str]

FSMap: TypeAlias = _FSMap | None
StoreLike: TypeAlias = Store | StorePath | FSMap | Path | str | dict[str, Buffer]

_P = ParamSpec('_P')
_R = TypeVar('_R')

class _Local(local):
    in_internal_call: bool = False
_local = _Local()
@contextmanager
def _enable_internal_calls():
    """Context manager to enable internal calls."""
    tmp = _local.in_internal_call
    _local.in_internal_call = True
    try:
        yield
    finally:
        _local.in_internal_call = tmp
def _require_internal_call(extra_msg: str = ''):
    """Decorator to require internal call context for method execution."""
    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not _local.in_internal_call:
                raise RuntimeError(
                    f'Calling `{func.__module__}.{func.__qualname__}`'
                    f' is only allowed inside internal calls. {extra_msg}'
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def _get_array(group: zarr.Group, path: str) -> zarr.Array:
    """Get a Zarr array from a group by path."""
    arr = group.get(path)
    if arr is None or not isinstance(arr, zarr.Array):
        raise KeyError(f'Array at path "{path}" not found in Zarr group.')
    return arr

def _get_group(group: zarr.Group, path: str) -> zarr.Group:
    """Get a Zarr group from a group by path."""
    grp = group.get(path)
    if grp is None or not isinstance(grp, zarr.Group):
        raise KeyError(f'Group at path "{path}" not found in Zarr group.')
    return grp

_Kvid = TypeVar('_Kvid', bound=str, covariant=True)
_Klm = TypeVar('_Klm', bound=str, covariant=True)
_Ktgt = TypeVar('_Ktgt', bound=str, covariant=True)

class DatasetItem(Generic[_Kvid, _Klm, _Ktgt]):
    """
    A container for a single dataset sample with videos, landmarks, and targets.

    This class holds normalized multimodal data for sign language recognition tasks.
    Use `from_sample()` factory method to create instances with automatic normalization.
    Direct instantiation via `__init__()` is restricted to internal use only.
    """

    @_require_internal_call(
        "Use `DatasetItem.from_sample` to create an instance."
    )
    def __init__(
        self,
        videos: Mapping[_Kvid,
            Tensor # shape: (T, C, H, W)
        ],
        landmarks: Mapping[_Klm, tuple[
            Tensor, # shape: (T, V, C=4) where C=[x, y, z, confidence]
            Tensor, # shape: (3, E) (from_, to_, weight)
        ]],
        connections: Mapping[tuple[_Klm, _Klm],
            Tensor # shape: (3, E) (from_, to_, weight)
        ],
        targets: Mapping[_Ktgt,
            Tensor # shape: (S)
        ]
        ):

        self.videos = videos
        "Mapping of video keys to video tensors. (N, T, C, H, W)."
        self.landmarks = landmarks
        "Mapping of landmark keys to (vertices, edges) tuples. (N, T, V, C=4)."
        self.connections = connections
        "Mapping of landmark pair keys to edge tensors. (N, 3, E)."
        self.targets = targets
        "Mapping of target keys to target tensors. (N, S)."

    @classmethod
    def from_sample(
        cls,
        videos: Mapping[_Kvid,
            Tensor # shape: (T, C, H, W) or (T, H, W)
        ],
        landmarks: Mapping[_Klm, tuple[
            Tensor, # shape: (T, V, C) where C=2,3,4 -> will be normalized to C=4 [x,y,z,conf]
            Tensor, # shape: (3, E) (from_, to_, weight)
        ]],
        connections: Mapping[tuple[_Klm, _Klm],
            Tensor # shape: (3, E) (from_, to_, weight)
        ],
        targets: Mapping[_Ktgt,
            Tensor # shape: (S?)
        ]
        ) -> 'DatasetItem[_Kvid, _Klm, _Ktgt]':
        """
        Create a DatasetItem from raw sample data with automatic normalization.
        
        This factory method normalizes and validates input tensors to ensure consistent
        formats across the dataset. It performs the following transformations:
        
        - Videos: Converts (T, H, W) to (T, C, H, W) with C=1 for grayscale images
        - Landmarks: Normalizes all landmarks to (T, V, C=4) format as [x, y, z, confidence]
          - C=2 input [x, y] → [x, y, 0, 1] (adds z=0, confidence=1)
          - C=3 input [x, y, z] → [x, y, z, 1] (adds confidence=1)
          - C=4 input [x, y, z, conf] → unchanged
        - Targets: Converts scalar tensors to 1D tensors
        
        Args:
            videos: Mapping of video keys to video tensors.
                Each tensor should have shape (T, H, W) or (T, C, H, W).
                Different videos can have different C, H, W dimensions.
            landmarks: Mapping of landmark keys to (vertices, edges) tuples.
                Vertices should have shape (T, V, C) where C is 2, 3, or 4.
                Different landmarks can have different V (number of vertices).
                Edges should have shape (3, E) containing [from, to, weight] for each edge.
            connections: Mapping of landmark pair keys to edge tensors.
                Each tensor should have shape (3, E) containing [from, to, weight].
            targets: Mapping of target keys to target tensors.
                Each tensor should be a 1D tensor of shape (S,) or a scalar.
                Different targets can have different sequence lengths S.
        
        Returns:
            DatasetItem: A normalized dataset item with:
                - videos: (T, C, H, W) format
                - landmarks: (T, V, C=4) format with C=[x, y, z, confidence]
                - connections: unchanged
                - targets: (S,) format
        
        Raises:
            ValueError: If input shapes are invalid or inconsistent T dimensions are detected.
        """

        with _enable_internal_calls():

            # ===== Process videos =====
            # Target format: (T, C, H, W)
            # - T: Must be consistent across all videos and landmarks
            # - C: Can vary across different video keys
            # - H: Can vary across different video keys
            # - W: Can vary across different video keys
            new_videos: dict[_Kvid, Tensor] = {}
            unique_shape_t: set[int] = set()

            for key, img in videos.items():
                match img.ndim:
                    case 3:
                        # (T, H, W) -> (T, 1, H, W)
                        img = img.unsqueeze(1)
                    case 4:
                        # Already (T, C, H, W)
                        pass
                    case _:
                        raise ValueError(
                            f'`videos[{key}]` must have shape (T, H, W) or (T, C, H, W),'
                            f' but got {img.shape}.'
                        )
                # Validate T dimension consistency across all videos
                unique_shape_t.add(img.shape[0])
                if len(unique_shape_t) > 1:
                    raise ValueError(
                        'All videos must have the same T dimension,'
                        f' but found multiple different T: {unique_shape_t}.'
                    )
                new_videos[key] = img

            # ===== Process landmarks =====
            # Target format: (T, V, C=4) where C represents [x, y, z, confidence]
            # - T: Must be consistent across all videos and landmarks
            # - V: Can vary across different landmark keys
            # - C: Fixed to 4 dimensions (x, y, z, confidence)
            #   Input can be (T, V, 2), (T, V, 3), or (T, V, 4)
            #   Missing z is filled with 0, missing confidence is filled with 1
            new_landmarks: dict[_Klm, tuple[Tensor, Tensor]] = {}
            for key, (v, e) in landmarks.items():
                # Step 1: Ensure 3D tensor (T, V, C)
                match v.ndim:
                    case 2:
                        # (T, V) -> (T, V, 1)
                        v = v.unsqueeze(-1)
                    case 3:
                        # Already (T, V, C)
                        pass
                    case _:
                        raise ValueError(
                            f'`landmarks[{key}][0]` must have shape (T, V) or (T, V, C),'
                            f' but got {v.shape}.'
                        )
                
                # Step 2: Normalize C dimension to 4 [x, y, z, confidence]
                C = v.shape[-1]
                if C < 2:
                    raise ValueError(
                        f'`landmarks[{key}][0]` must have C dimension of at least 2 (x, y),'
                        f' but got C={C} with shape {v.shape}.'
                    )
                elif C > 4:
                    raise ValueError(
                        f'`landmarks[{key}][0]` must have C dimension of at most 4 (x, y, z, confidence),'
                        f' but got C={C} with shape {v.shape}.'
                    )
                
                # Step 2-1: C=2 -> C=3 (add z=0)
                if v.shape[-1] == 2:
                    # (T, V, 2) [x, y] -> (T, V, 3) [x, y, 0]
                    z_padding = torch.zeros(v.shape[0], v.shape[1], 1, dtype=v.dtype, device=v.device)
                    v = torch.cat([v, z_padding], dim=-1)
                
                # Step 2-2: C=3 -> C=4 (add confidence=1)
                if v.shape[-1] == 3:
                    # (T, V, 3) [x, y, z] -> (T, V, 4) [x, y, z, 1]
                    conf_padding = torch.ones(v.shape[0], v.shape[1], 1, dtype=v.dtype, device=v.device)
                    v = torch.cat([v, conf_padding], dim=-1)
                
                # Validate T dimension consistency with videos
                unique_shape_t.add(v.shape[0])
                if len(unique_shape_t) > 1:
                    raise ValueError(
                        'All landmarks and videos must have the same T dimension,'
                        f' but found multiple different T: {unique_shape_t}.'
                    )
                new_landmarks[key] = (v, e)

            # ===== Process targets =====
            # Target format: (S,)
            # - S: Can vary across different target keys
            new_targets: dict[_Ktgt, Tensor] = {}
            for key, tgt in targets.items():
                match tgt.ndim:
                    case 0:
                        # Scalar -> (1,)
                        tgt = tgt.unsqueeze(0)
                    case 1:
                        # Already (S,)
                        pass
                    case _:
                        raise ValueError(
                            f'`targets[{key}]` must have shape (S,) or (),'
                            f' but got {tgt.shape}.'
                        )
                new_targets[key] = tgt

            # Finally, create the DatasetItem instance
            return cls(
                videos=new_videos,
                landmarks=new_landmarks,
                connections=connections,
                targets=new_targets,
            )

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> Self:
        """Create a DatasetItem from a Zarr group.

        Args:
            group (zarr.Group): The Zarr group to read data from.

        Returns:
            Self: A DatasetItem instance.
        """

        videos: dict[_Kvid, Tensor] = {}
        landmarks_landmarks: dict[_Klm, Tensor] = {}
        landmarks_connections: dict[_Klm, Tensor] = {}
        connections: dict[tuple[_Klm, _Klm], Tensor] = {}
        targets: dict[_Ktgt, Tensor] = {}

        for karr in group.array_keys():
            splited_key = karr.split('.')
            arr = _get_array(group, karr)
            if karr.startswith('videos.'):
                kvid = cast(_Kvid, splited_key[1])
                videos[kvid] = torch.tensor(arr[:])
            elif karr.startswith('landmarks.landmarks'):
                klm = cast(_Klm, splited_key[2])
                landmarks_landmarks[klm] = torch.tensor(arr[:])
            elif karr.startswith('landmarks.connections'):
                klm = cast(_Klm, splited_key[2])
                landmarks_connections[klm] = torch.tensor(arr[:])
            elif karr.startswith('connections.'):
                klm1 = cast(_Klm, splited_key[1])
                klm2 = cast(_Klm, splited_key[2])
                connections[(klm1, klm2)] = torch.tensor(arr[:])
            elif karr.startswith('targets.'):
                ktgt = cast(_Ktgt, splited_key[1])
                targets[ktgt] = torch.tensor(arr[:])
            else:
                raise KeyError(f'Unknown array key "{karr}" in Zarr group.')

        common_keys = landmarks_landmarks.keys() ^ landmarks_connections.keys()

        with _enable_internal_calls():
            return cls(
                videos=videos,
                landmarks={
                    klm: (landmarks_landmarks[klm], landmarks_connections[klm])
                    for klm in common_keys
                },
                connections=connections,
                targets=targets,
            )

    def to_zarr(self, group: zarr.Group):
        """Write the dataset to a Zarr group.

        Args:
            group (zarr.Group): The Zarr group to write data to.
        """

        for kvid, vid in self.videos.items():
            group.create_array(
                f'videos.{kvid}',
                data=vid.cpu().numpy(),
            )
        for klm, (lm, conn) in self.landmarks.items():
            group.create_array(
                f'landmarks.landmarks.{klm}',
                data=lm.cpu().numpy(),
            )
            group.create_array(
                f'landmarks.connections.{klm}',
                data=conn.cpu().numpy(),
            )
        for (klm1, klm2), conn in self.connections.items():
            group.create_array(
                f'connections.{klm1}.{klm2}',
                data=conn.cpu().numpy(),
            )
        for ktgt, tgt in self.targets.items():
            group.create_array(
                f'targets.{ktgt}',
                data=tgt.cpu().numpy(),
            )

    @classmethod
    def from_folder(cls, dirpath: PathLike) -> Self:
        """
        Create a DatasetItem from a folder of files.

        Args:
            dirpath (PathLike): Path to the folder containing the dataset files.

        Returns:
            Self: A DatasetItem instance.
        """

        dirpath = Path(dirpath)

        videos: dict[_Kvid, Tensor] = {}
        landmarks_landmarks: dict[_Klm, Tensor] = {}
        landmarks_connections: dict[_Klm, Tensor] = {}
        connections: dict[tuple[_Klm, _Klm], Tensor] = {}
        targets: dict[_Ktgt, Tensor] = {}

        for filepath in dirpath.iterdir():
            splited_name = filepath.name.split('.')
            if filepath.name.startswith('video.'):
                kvid = cast(_Kvid, splited_name[1])
                videos[kvid] = torch.load(filepath)
            elif filepath.name.startswith('landmark.landmark.'):
                klm = cast(_Klm, splited_name[2])
                landmarks_landmarks[klm] = torch.load(filepath)
            elif filepath.name.startswith('landmark.connection.'):
                klm = cast(_Klm, splited_name[2])
                landmarks_connections[klm] = torch.load(filepath)
            elif filepath.name.startswith('connection.'):
                klm1 = cast(_Klm, splited_name[1])
                klm2 = cast(_Klm, splited_name[2])
                connections[(klm1, klm2)] = torch.load(filepath)
            elif filepath.name.startswith('target.'):
                ktgt = cast(_Ktgt, splited_name[1])
                targets[ktgt] = torch.load(filepath)
            else:
                pass # Ignore unknown files

        common_keys = landmarks_landmarks.keys() ^ landmarks_connections.keys()

        with _enable_internal_calls():
            return cls(
                videos=videos,
                landmarks={
                    klm: (landmarks_landmarks[klm], landmarks_connections[klm])
                    for klm in common_keys
                },
                connections=connections,
                targets=targets,
            )

    def to_folder(
        self, dirpath: PathLike
        ):
        """
        Export the dataset to a filesystem directory.

        Creates a directory structure with metadata JSON and numbered item subdirectories.
        Each dataset item is serialized via DatasetItem.to_folder().

        Args:
            dirpath (PathLike): Target directory path (created if not exists).
        """

        for kvid, vid in self.videos.items():
            torch.save(
                vid,
                Path(dirpath) / f'video.{kvid}.pt'
            )
        for klm, (lm, conn) in self.landmarks.items():
            torch.save(
                lm,
                Path(dirpath) / f'landmark.{klm}.landmark.pt'
            )
            torch.save(
                conn,
                Path(dirpath) / f'landmark.{klm}.connection.pt'
            )
        for (klm1, klm2), conn in self.connections.items():
            torch.save(
                conn,
                Path(dirpath) / f'connection.{klm1}.{klm2}.pt'
            )
        for ktgt, tgt in self.targets.items():
            torch.save(
                tgt,
                Path(dirpath) / f'target.{ktgt}.pt'
            )

    # lightning.pytorch.utilities.move_data_to_device calls this method
    def to(self, device: torch.device | str) -> Self:
        """
        Move all tensors to the specified device.
        
        This method is called by PyTorch Lightning's move_data_to_device utility.
        
        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        
        Returns:
            Self: A new DatasetItem with all tensors on the specified device.
        """
        new_videos = {
            key: vid.to(device)
            for key, vid in self.videos.items()
        }
        new_landmarks = {
            key: (lm.to(device), conn.to(device))
            for key, (lm, conn) in self.landmarks.items()
        }
        new_connections = {
            key: conn.to(device)
            for key, conn in self.connections.items()
        }
        new_targets = {
            key: tgt.to(device)
            for key, tgt in self.targets.items()
        }
        with _enable_internal_calls():
            return type(self)(
                videos=new_videos,
                landmarks=new_landmarks,
                connections=new_connections,
                targets=new_targets,
            )

def collate_fn(
    batch: list[DatasetItem[_Kvid, _Klm, _Ktgt]]
    ) -> DatasetItem[_Kvid, _Klm, _Ktgt]:
    """
    Collate a batch of dataset items into a single item.

    Args:
        batch: List of dataset items to collate.

    Returns:
        DatasetItem: A single collated dataset item.
    """

    first_item = batch[0]

    new_videos = {
        key: torch.cat([
            item.videos[key].unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)
            for item in batch
        ], dim=0)  # shape: (N, T, C, H, W)
        for key in first_item.videos.keys()
    }

    new_landmarks = {
        key: (
            torch.nn.utils.rnn.pad_sequence([
                item.landmarks[key][0]
                for item in batch
            ], batch_first=True), # shape: (N, T, V, C=4)
            first_item.landmarks[key][1]
        )
        for key in first_item.landmarks.keys()
    }

    new_targets = {
        key: torch.nn.utils.rnn.pad_sequence([
            item.targets[key]
            for item in batch
        ], batch_first=True) # shape: (N, S)
        for key in first_item.targets.keys()
    }

    with _enable_internal_calls():
        return DatasetItem(
            videos=new_videos,
            landmarks=new_landmarks,
            connections=first_item.connections,
            targets=new_targets,
        )

class Dataset(ABC, _Dataset[DatasetItem[_Kvid, _Klm, _Ktgt]]):

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The number of samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> DatasetItem[_Kvid, _Klm, _Ktgt]:
        """
        Retrieve a single sample from the dataset.
        
        Args:
            index: The index of the sample to retrieve (0-based).
        
        Returns:
            DatasetItem: A dataset item containing videos, landmarks, connections, and targets.
        """
        pass

class IterableDataset(ABC, _IterableDataset[DatasetItem[_Kvid, _Klm, _Ktgt]]):
    """
    Abstract base class for iterable-style datasets returning DatasetItem.
    Requires implementing __iter__ for sequential access without indexing.
    Useful for streaming large datasets or data from external sources.
    Compatible with PyTorch DataLoader for batching and multiprocessing.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetItem[_Kvid, _Klm, _Ktgt]]:
        """
        Return an iterator over the dataset samples.
        
        Yields:
            DatasetItem: Dataset items one at a time.
        """
        pass

######################## Utility Dataset Classes #########################

class CacheItemDataset(Dataset[_Kvid, _Klm, _Ktgt]):
    """
    Caching wrapper for map-style datasets to avoid redundant data loading.
    Stores loaded items in memory for fast repeated access.
    Useful for small to medium datasets that fit in RAM.
    Call clear_cache() to free memory when needed.

    Args:
        base_dataset: The underlying Dataset to wrap and cache items from.
    """

    def __init__(self, base_dataset: Dataset[_Kvid, _Klm, _Ktgt]):
        self._base_dataset = base_dataset
        self._cache: dict[int, DatasetItem[_Kvid, _Klm, _Ktgt]] = {}

    def clear_cache(self):
        """Clear the internal cache to free memory."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, index: int) -> DatasetItem[_Kvid, _Klm, _Ktgt]:
        if index in self._cache:
            return self._cache[index]
        item = self._base_dataset[index]
        self._cache[index] = item
        return item


############################ Zarr Integration ############################

def dataset_to_zarr(
    dataset: (
        Dataset[_Kvid, _Klm, _Ktgt] |
        IterableDataset[_Kvid, _Klm, _Ktgt]
    ),
    store: StoreLike,
    **metadata: JSON,
    ):
    """
    Export a dataset to Zarr format for efficient storage and retrieval.
    
    Creates a Zarr group hierarchy with metadata and numbered item groups.
    Each dataset item is serialized via DatasetItem.to_zarr().
    Supports both map-style and iterable-style datasets.
    
    Args:
        dataset: Source dataset (Dataset or IterableDataset).
        store: Zarr storage location (path, Store, StorePath, etc.).
        **metadata: Additional metadata stored in 'metadata' group attributes.
    
    Structure:
        {store}/metadata/  - Group with metadata attributes
        {store}/0/         - First item group
        {store}/1/         - Second item group
        ...
    """

    root = zarr.create_group(store)

    root.create_group('metadata', **metadata)

    if isinstance(dataset, Dataset):
        iterator = enumerate(dataset[i] for i in range(len(dataset)))
    else:
        iterator = enumerate(dataset)

    for idx, item in iterator:
        item_group = root.create_group(str(idx))
        item.to_zarr(item_group)

class ZarrDataset(Dataset[_Kvid, _Klm, _Ktgt]):
    """
    Read-only dataset backed by Zarr storage for efficient array access.
    Supports local filesystem and cloud storage via fsspec.
    Metadata is stored in a dedicated 'metadata' group.
    Items are indexed by numeric keys (0, 1, 2, ...).
    Efficient for large datasets with chunked array storage.

    Args:
        store: Zarr storage location (path, Store, StorePath, etc.)
    """

    def __init__(self, store: StoreLike):
        self._root = zarr.open_group(store, mode='r')

    @property
    def metadata(self) -> Mapping[str, Any]:
        metadata_group = _get_group(self._root, 'metadata')
        return dict(metadata_group.attrs)

    def __len__(self) -> int:
        return sum(1 for _ in self._root.group_keys()) - 1

    def __getitem__(self, index: int) -> DatasetItem[_Kvid, _Klm, _Ktgt]:
        item_group = _get_group(self._root, str(index))
        return DatasetItem[_Kvid, _Klm, _Ktgt].from_zarr(item_group)

def dataset_from_zarr(
    store: StoreLike
    ) -> ZarrDataset[Any, Any, Any]:
    """Create a ZarrDataset from a Zarr store."""
    return ZarrDataset(store)

######################## File System Integration #########################

def dataset_to_folder(
    dataset: (
        Dataset[_Kvid, _Klm, _Ktgt] |
        IterableDataset[_Kvid, _Klm, _Ktgt]
    ),
    dirpath: PathLike,
    **metadata: JSON,
    ):
    """
    Export a dataset to a filesystem directory for portable storage.
    
    Creates a directory structure with metadata JSON and numbered item subdirectories.
    Each dataset item is serialized via DatasetItem.to_folder().
    Supports both map-style and iterable-style datasets.
    
    Args:
        dataset: Source dataset (Dataset or IterableDataset).
        dirpath: Target directory path (created if not exists).
        **metadata: Additional metadata saved to metadata.json.
    
    Structure:
        {dirpath}/metadata.json  - Metadata JSON file
        {dirpath}/items/0/       - First item directory
        {dirpath}/items/1/       - Second item directory
        ...
    """

    import json

    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    json.dump(metadata, (dirpath / 'metadata.json').open('w'))

    items_dir = dirpath / 'items'
    items_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, Dataset):
        iterator = enumerate(dataset[i] for i in range(len(dataset)))
    else:
        iterator = enumerate(dataset)

    for idx, item in iterator:
        item_dir = items_dir / str(idx)
        item_dir.mkdir(parents=True, exist_ok=True)
        item.to_folder(item_dir)

class FileSystemDataset(Dataset[_Kvid, _Klm, _Ktgt]):
    """
    Dataset backed by filesystem directory structure for portability.
    Each item is stored in a numbered subdirectory (0, 1, 2, ...).
    Metadata is stored in metadata.json at the root level.
    Supports any filesystem accessible via pathlib.Path.
    Useful for human-readable and debuggable storage.

    Args:
        dirpath: Root directory path containing dataset structure.
    """

    def __init__(self, dirpath: PathLike):
        import json

        dirpath = Path(dirpath)
        self._dirpath = dirpath

        metadata_path = dirpath / 'metadata.json'
        if metadata_path.exists():
            with metadata_path.open('r') as f:
                self._metadata: Mapping[str, Any] = json.load(f)
        else:
            self._metadata = {}

        self._items_dir = dirpath / 'items'

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return sum(1 for _ in self._items_dir.iterdir() if _.is_dir())

    def __getitem__(self, index: int) -> DatasetItem[_Kvid, _Klm, _Ktgt]:
        item_dir = self._items_dir / str(index)
        return DatasetItem[_Kvid, _Klm, _Ktgt].from_folder(item_dir)

def dataset_from_folder(
    dirpath: PathLike
    ) -> FileSystemDataset[Any, Any, Any]:
    """Create a FileSystemDataset from a directory path."""
    return FileSystemDataset(dirpath)

##################### PyTorch DataLoader Integration #####################

def create_dataloader(
    dataset: (
        Dataset[_Kvid, _Klm, _Ktgt] | 
        IterableDataset[_Kvid, _Klm, _Ktgt] 
    ),
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable[[int], None] | None = None,
    mp_context: str | BaseContext | None = None,
    generator: torch.Generator | None = None,
    *,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    ) -> DataLoader[DatasetItem[_Kvid, _Klm, _Ktgt]]:
    """Create a PyTorch DataLoader for the given dataset.

    This function wraps the dataset in a DataLoader, allowing for easy batching,
    shuffling, and parallel data loading.

    Args:
        dataset: The dataset to load data from (Dataset or IterableDataset).
        batch_size: Number of samples per batch to load (default: 1).
        shuffle: Whether to shuffle the data at every epoch (default: False).
        num_workers: Number of subprocesses to use for data loading (default: 0).
        drop_last: Whether to drop the last incomplete batch (default: False).
        timeout: Timeout for collecting a batch from workers (default: 0).
        worker_init_fn: Function to initialize each worker process (default: None).
        mp_context: Multiprocessing context to use (default: None).
        generator: Random number generator for shuffling (default: None).
        prefetch_factor: Number of samples loaded in advance by each worker (default: None).
        persistent_workers: Whether to keep workers alive between epochs (default: False).

    Returns:
        DataLoader: A PyTorch DataLoader instance for the dataset.

    References:
        1. PyTorch DataLoader documentation:
           https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    """

    return DataLoader( # pyright: ignore[reportUnknownVariableType]
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp_context,
        generator=generator,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
