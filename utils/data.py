
import itertools
import random
import numpy as np
import pandas as pd
import torchvision

import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, Dataset
from typing import TypeVar, Iterable, Dict, List

from typing import Callable

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


def send_to_device(tensor, device):

    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


class ImbalancedDatasetSampler(Sampler):

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class RandomMultipleGallerySampler(Sampler):


    def __init__(self, dataset, num_instances=4):
        super(RandomMultipleGallerySampler, self).__init__(dataset)
        self.dataset = dataset
        self.num_instances = num_instances

        self.idx_to_pid = {}
        self.cid_list_per_pid = {}
        self.idx_list_per_pid = {}

        for idx, (_, pid, cid) in enumerate(dataset):
            if pid not in self.cid_list_per_pid:
                self.cid_list_per_pid[pid] = []
                self.idx_list_per_pid[pid] = []

            self.idx_to_pid[idx] = pid
            self.cid_list_per_pid[pid].append(cid)
            self.idx_list_per_pid[pid].append(idx)

        self.pid_list = list(self.idx_list_per_pid.keys())
        self.num_samples = len(self.pid_list)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        def select_idxes(element_list, target_element):
            assert isinstance(element_list, list)
            return [i for i, element in enumerate(element_list) if element != target_element]

        pid_idxes = torch.randperm(len(self.pid_list)).tolist()
        final_idxes = []

        for perm_id in pid_idxes:
            i = random.choice(self.idx_list_per_pid[self.pid_list[perm_id]])
            _, _, cid = self.dataset[i]

            final_idxes.append(i)

            pid_i = self.idx_to_pid[i]
            cid_list = self.cid_list_per_pid[pid_i]
            idx_list = self.idx_list_per_pid[pid_i]
            selected_cid_list = select_idxes(cid_list, cid)

            if selected_cid_list:
                if len(selected_cid_list) >= self.num_instances:
                    cid_idxes = np.random.choice(selected_cid_list, size=self.num_instances - 1, replace=False)
                else:
                    cid_idxes = np.random.choice(selected_cid_list, size=self.num_instances - 1, replace=True)
                for cid_idx in cid_idxes:
                    final_idxes.append(idx_list[cid_idx])
            else:
                selected_idxes = select_idxes(idx_list, i)
                if not selected_idxes:
                    continue
                if len(selected_idxes) >= self.num_instances:
                    pid_idxes = np.random.choice(selected_idxes, size=self.num_instances - 1, replace=False)
                else:
                    pid_idxes = np.random.choice(selected_idxes, size=self.num_instances - 1, replace=True)

                for pid_idx in pid_idxes:
                    final_idxes.append(idx_list[pid_idx])

        return iter(final_idxes)


class CombineDataset(Dataset[T_co]):


    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(CombineDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        return list(itertools.chain(*[d[idx] for d in self.datasets]))


def concatenate(tensors):

    if isinstance(tensors[0], torch.Tensor):
        return torch.cat(tensors, dim=0)
    elif isinstance(tensors[0], List):
        ret = []
        for i in range(len(tensors[0])):
            ret.append(concatenate([t[i] for t in tensors]))
        return ret
    elif isinstance(tensors[0], Dict):
        ret = dict()
        for k in tensors[0].keys():
            ret[k] = concatenate([t[k] for t in tensors])
        return ret
