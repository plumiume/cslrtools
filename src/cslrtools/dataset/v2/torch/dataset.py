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

    @_require_internal_call(
        "Use `DatasetItem.from_sample` to create an instance."
    )
    def __init__(
        self,
        videos: Mapping[_Kvid,
            Tensor # shape: (T, H, W, C)
        ],
        landmarks: Mapping[_Klm, tuple[
            Tensor, # shape: (T, V, C)
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
        self.landmarks = landmarks
        self.connections = connections
        self.targets = targets

    @classmethod
    def from_sample(
        cls,
        videos: Mapping[_Kvid,
            Tensor # shape: (T, H, W, C?)
        ],
        landmarks: Mapping[_Klm, tuple[
            Tensor, # shape: (T, V, C?)
            Tensor, # shape: (3, E) (from_, to_, weight)
        ]],
        connections: Mapping[tuple[_Klm, _Klm],
            Tensor # shape: (3, E) (from_, to_, weight)
        ],
        targets: Mapping[_Ktgt,
            Tensor # shape: (S?)
        ]
        ) -> 'DatasetItem[_Kvid, _Klm, _Ktgt]':

        with _enable_internal_calls():

            new_videos: dict[_Kvid, Tensor] = {}
            unique_shape_t: set[int] = set()

            for key, img in videos.items():
                match img.ndim:
                    case 3:
                        # (T, H, W) -> (T, H, W, 1)
                        img = img.unsqueeze(-1)
                    case 4:
                        pass
                    case _:
                        raise ValueError(
                            f'`videos[{key}]` must have shape (T, H, W) or (T, H, W, C),'
                            f' but got {img.shape}.'
                        )
                unique_shape_t.add(img.shape[0])
                if len(unique_shape_t) > 1:
                    raise ValueError(
                        'All videos must have the same T dimension,'
                        f' but found multiple different T: {unique_shape_t}.'
                    )
                new_videos[key] = img

            new_landmarks: dict[_Klm, tuple[Tensor, Tensor]] = {}
            unique_shape_v_c: set[tuple[int, int]] = set()
            for key, (v, e) in landmarks.items():
                match v.ndim:
                    case 2:
                        # (T, V) -> (T, V, 1)
                        v = v.unsqueeze(-1)
                    case 3:
                        pass
                    case _:
                        raise ValueError(
                            f'`landmarks[{key}][0]` must have shape (T, V) or (T, V, C),'
                            f' but got {v.shape}.'
                        )
                unique_shape_v_c.add((v.shape[-2], v.shape[-1]))
                if len(unique_shape_v_c) > 1:
                    raise ValueError(
                        'All landmarks must have the same (V, C) shape,'
                        f' but found multiple different shapes: {unique_shape_v_c}.'
                    )
                unique_shape_t.add(v.shape[0])
                if len(unique_shape_t) > 1:
                    raise ValueError(
                        'All landmarks and videos must have the same T dimension,'
                        f' but found multiple different T: {unique_shape_t}.'
                    )
                new_landmarks[key] = (v, e)

            new_targets: dict[_Ktgt, Tensor] = {}
            for key, tgt in targets.items():
                match tgt.ndim:
                    case 0:
                        # ( ) -> (1)
                        tgt = tgt.unsqueeze(0)
                    case 1:
                        pass
                    case _:
                        raise ValueError(
                            f'`targets[{key}]` must have shape (S,) or (),'
                            f' but got {tgt.shape}.'
                        )
                new_targets[key] = tgt

            return cls(
                videos=new_videos,
                landmarks=new_landmarks,
                connections=connections,
                targets=new_targets,
            )

    # lightning.pytorch.utilities.move_data_to_device calls this method
    def to(self, device: torch.device) -> Self:
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
        key: torch.nn.utils.rnn.pad_sequence([
            item.videos[key]
            for item in batch
        ], batch_first=True) # shape: (B, T, H, W, C)
        for key in first_item.videos.keys()
    }

    new_landmarks = {
        key: (
            torch.nn.utils.rnn.pad_sequence([
                item.landmarks[key][0]
                for item in batch
            ], batch_first=True), # shape: (B, T, V, C)
            first_item.landmarks[key][1]
        )
        for key in first_item.landmarks.keys()
    }

    new_targets = {
        key: torch.nn.utils.rnn.pad_sequence([
            item.targets[key]
            for item in batch
        ], batch_first=True) # shape: (B, S)
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
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> DatasetItem[_Kvid, _Klm, _Ktgt]:
        pass

class IterableDataset(ABC, _IterableDataset[DatasetItem[_Kvid, _Klm, _Ktgt]]):

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetItem[_Kvid, _Klm, _Ktgt]]:
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
