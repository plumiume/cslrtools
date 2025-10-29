from abc import ABC, abstractmethod
from typing import cast, Any, Literal, TypeVar, Generic, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset as _Dataset

PaddingMode = Literal['constant', 'reflect', 'replicate', 'circular']

_M = TypeVar("_M", bound=Mapping[str, Any])

@dataclass
class DatasetItem(Generic[_M]):
    x: Tensor # [T: dynamic, V: fixed, C: fixed]
    y: Tensor # [S: dynamic]
    meta: _M = cast(_M, {})

_Item = TypeVar("_Item", bound=DatasetItem[Mapping[str, Any]])

class CollateFn(Generic[_Item]):

    def __init__(
        self,
        pad_mode: PaddingMode = 'constant',
        pad_value: float = 0.0,
        blank: int = 0,
        x_maxlen: int | None = None,
        y_maxlen: int | None = None,
        ):

        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.blank = blank
        self.x_maxlen = x_maxlen
        self.y_maxlen = y_maxlen

    def __call__(self, batch: list[_Item]) -> DatasetItem[dict[str, list[Any]]]:

        x_maxlen = self.x_maxlen or max(item.x.shape[0] for item in batch)
        y_maxlen = self.y_maxlen or max(item.y.shape[0] for item in batch)

        return DatasetItem(
            x=torch.stack([
                pad(
                    item.x,
                    (0, 0, 0, 0, 0, x_maxlen - item.x.shape[-3]),
                    mode=self.pad_mode,
                    value=self.pad_value
                )
                for item in batch
            ]),
            y=torch.stack([
                pad(
                    item.y,
                    (0, y_maxlen - item.y.shape[-1]),
                    mode='constant',
                    value=self.blank
                )
                for item in batch
            ]),
            meta={
                key: [item.meta.get(key) for item in batch]
                for key in batch[0].meta.keys()
            }
        )

class Dataset(_Dataset[_Item], ABC):

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> _Item: ...

