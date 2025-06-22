# Copyright 2025 plumiume.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import lightning.pytorch
except ImportError:
    raise ImportError(
        "lightning.pytorch is required to use LightningDataModule. "
        "Please install it with 'pip install .[lightning]'"
    )

from typing import Literal, TypedDict, Generic
import torch
from .pytorch import Dataset, _M

StageString = Literal['fit', 'val', 'test', 'predict']

class LightningDataModule(lightning.pytorch.LightningDataModule, Generic[_M]):

    class DataLoaderCommonKwargs(TypedDict, total=False):
        batch_size: int
        num_workers: int
        pin_memory: bool
        persistent_workers: bool

    class DataLoaderKwargs(TypedDict):
        shuffle: bool
        drop_last: bool

    train_kwargs = DataLoaderKwargs({'shuffle': True, 'drop_last': True})
    val_kwargs = DataLoaderKwargs({'shuffle': False, 'drop_last': False})
    test_kwargs = val_kwargs
    predict_kwargs = val_kwargs

    @property
    def batch_size(self) -> int | None:
        return self.common_kwargs.get('batch_size', None)
    @batch_size.setter
    def batch_size(self, value: int):
        self.common_kwargs['batch_size'] = value

    def __init__(
        self,
        dataset: Dataset[_M],
        stages: list[list[StageString]],
        filters: list[bool] | None = None,
        common_kwargs: DataLoaderCommonKwargs = DataLoaderCommonKwargs({})
        ):
        super().__init__()
        if len(dataset) != len(stages):
            raise ValueError(
                f"Dataset length {len(dataset)} does not match stages length {len(stages)}."
            )
        self.dataset = dataset
        self.stages = stages
        self.filters = [True for _ in stages] if filters is None else filters
        self.common_kwargs = common_kwargs

    def _setup_subset(self, stage: StageString):
        indices = [i for i, (s, f) in enumerate(zip(self.stages, self.filters)) if stage in s and f]
        return torch.utils.data.Subset(self.dataset, indices)

    def setup(self, stage: StageString | str | None = None):
        match stage:
            case 'fit':
                self._fit_dataset = self._setup_subset('fit')
                self._val_dataset = self._setup_subset('val')
            case 'validate':
                self._val_dataset = self._setup_subset('val')
            case 'test':
                self._test_dataset = self._setup_subset('test')
            case 'predict':
                self._predict_dataset = self._setup_subset('predict')
            case _:
                pass

    def train_dataloader(self):
        # fit
        return torch.utils.data.DataLoader(
            self._fit_dataset,
            **self.common_kwargs, **self.train_kwargs
        )

    def val_dataloader(self):
        # fit and validate
        return torch.utils.data.DataLoader(
            self._val_dataset,
            **self.common_kwargs, **self.val_kwargs
        )

    def test_dataloader(self):
        # test
        return torch.utils.data.DataLoader(
            self._test_dataset,
            **self.common_kwargs, **self.test_kwargs
        )

    def predict_dataloader(self):
        # predict
        return torch.utils.data.DataLoader(
            self._predict_dataset,
            **self.common_kwargs, **self.predict_kwargs
        )