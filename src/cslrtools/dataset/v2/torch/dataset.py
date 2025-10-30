from abc import ABC, abstractmethod
from typing import (
    TypeVar, ParamSpec, Generic,
    Hashable, Mapping,
    Callable, Self, Iterator,
)

from functools import wraps
from contextlib import contextmanager
from threading import local
from multiprocessing.context import BaseContext

import torch
from torch import Tensor
from torch.utils.data import (
    Dataset as _Dataset,
    IterableDataset as _IterableDataset,
    DataLoader
)

_P = ParamSpec('_P')
_R = TypeVar('_R')

class _Local(local):
    in_internal_call: bool = False
_local = _Local()
@contextmanager
def _enable_internal_calls():
    tmp = _local.in_internal_call
    _local.in_internal_call = True
    try:
        yield
    finally:
        _local.in_internal_call = tmp
def _require_internal_call(extra_msg: str = ''):
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

_Kvid = TypeVar('_Kvid', bound=Hashable, covariant=True)
_Klm = TypeVar('_Klm', bound=Hashable, covariant=True)
_Ktgt = TypeVar('_Ktgt', bound=Hashable, covariant=True)

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

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetItem[_Kvid, _Klm, _Ktgt]]:
        """
        Return an iterator over the dataset samples.
        
        Yields:
            DatasetItem: Dataset items one at a time.
        """
        pass

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
    # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

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
