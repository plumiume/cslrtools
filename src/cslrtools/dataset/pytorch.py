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

from typing import (
    SupportsIndex, NamedTuple, TypedDict,
    TypeVar, Generic
)
from itertools import chain
import torch
from torch import Tensor

class DataTuple(NamedTuple):
    input: Tensor # (N?, T, D)
    input_len: Tensor # (N?,)
    label: Tensor # (N?, S)
    label_len: Tensor # (N?,)

class Metadata(TypedDict, total=False): ...

_M = TypeVar('_M', bound=Metadata)

class Dataset(torch.utils.data.Dataset[DataTuple], Generic[_M]):

    def __init__(
        self,
        inputs: list[Tensor],
        input_maxlen: int,
        labels: list[Tensor],
        label_maxlen: int,
        blank_label: str,
        blank_idx: int,
        inputs_mean: Tensor,
        inputs_var: Tensor,
        classes: dict[str, int],
        classes_inv: dict[int, str],
        metas: list[_M] = []
        ):

        self.inputs = inputs
        self.input_maxlen = input_maxlen
        self.labels = labels
        self.label_maxlen = label_maxlen
        self._blank_label = blank_label
        self._blank_idx = blank_idx
        self._inputs_mean = inputs_mean
        self._inputs_var = inputs_var
        self._classes = classes
        self._classes_inv = classes_inv
        self._metas = metas

    @classmethod
    def from_sequences(
        cls,
        inputs: list[Tensor],  # list of (T, D)
        labels: list[list[str]],  # list of label sequences (S,)
        blank_label: str = ' ',
        metas: list[_M] = []
    ) -> 'Dataset':

        sample_lens = list[Tensor]()
        inputs_mean = list[Tensor]()
        inputs_var = list[Tensor]()

        for xi in inputs:
            xi = xi.nan_to_num(posinf=float('nan'), neginf=float('nan'))
            
            xi_len = (~xi.isnan()).sum(0)
            xi_mean = xi.nanmean(0)
            xi_var = ((xi - xi_mean) ** 2).nanmean(0)

            sample_lens.append(xi_len)
            inputs_mean.append(xi_mean)
            inputs_var.append(xi_var)

        sample_lens = torch.stack(sample_lens)
        inputs_mean = torch.stack(inputs_mean)
        inputs_var = torch.stack(inputs_var)

        sample_lens_sum = sample_lens.sum(0)
        inputs_mean = (inputs_mean * sample_lens).nansum(0) / sample_lens_sum
        inputs_var = (inputs_var * sample_lens).nansum(0) / sample_lens_sum


        labels_set = {blank_label} | set(chain.from_iterable(labels))
        ordered_labels = sorted(labels_set)
        classes_inv = dict(enumerate(ordered_labels))
        classes = {v: k for k, v in classes_inv.items()}
        blank_idx = classes[blank_label]

        inputs_tensor = [
            (x - inputs_mean) / inputs_var.sqrt() for x in inputs
        ]
        input_maxlen = max((len(x) for x in inputs), default=0)

        labels_tensor = [
            torch.tensor([classes[l] for l in label_seq])
            for label_seq in labels
        ]
        label_maxlen = max((len(y) for y in labels), default=0)

        return cls(
            inputs=inputs_tensor,
            input_maxlen=input_maxlen,
            labels=labels_tensor,
            label_maxlen=label_maxlen,
            blank_label=blank_label,
            blank_idx=blank_idx,
            inputs_mean=inputs_mean,
            inputs_var=inputs_var,
            classes=classes,
            classes_inv=classes_inv,
            metas=metas
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: SupportsIndex) -> DataTuple:
        input = self.inputs[idx]
        label = self.labels[idx]
        return DataTuple(
            input=torch.nn.functional.pad(
                input, (0, 0, 0, self.input_maxlen - input.shape[-2]),
                value=0
            ),
            input_len=torch.tensor(input.shape[0], device=input.device),
            label=torch.nn.functional.pad(
                label, (0, self.label_maxlen - label.shape[-1]),
                value=self._blank_idx
            ),
            label_len=torch.tensor(label.shape[0], device=label.device)
        )

    @property
    def blank_idx(self) -> int:
        return self._blank_idx

    @property
    def blank_label(self) -> str:
        return self._blank_label

    @property
    def classes(self) -> list[str]:
        return list(self._classes.keys())

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def num_features(self) -> int:
        return self.inputs[0].shape[1] if self.inputs else 0

    def scaling(self, x: Tensor) -> Tensor:
        return (x - self._inputs_mean) / self._inputs_var.sqrt()
    def inv_scaling(self, x: Tensor) -> Tensor:
        return x * self._inputs_var.sqrt() + self._inputs_mean
    def label_to_idx(self, label: str) -> int:
        return self._classes[label]
    def inv_label_to_idx(self, idx: int) -> str:
        return self._classes_inv[idx]
