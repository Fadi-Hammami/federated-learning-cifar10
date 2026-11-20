"""
Federated CIFAR-10 dataset (in-memory) built on FedLab's CIFAR10Partitioner.

Partition scenarios (argument `partition`):
  - "iid"           : balanced, every client gets a random class-balanced shard
                      (FedLab: balance=True, partition="iid")
  - "noniid-shards" : pathological non-IID of the FedAvg paper: data sorted by label,
                      cut into 2*num_clients shards, 2 shards per client -> ~2 classes/client
                      (FedLab: balance=None, partition="shards", num_shards=2*num_clients)
  - "noniid-dir"    : Dirichlet label skew with concentration `dir_alpha`
                      (FedLab: balance=None, partition="dirichlet", dir_alpha=alpha)

Data loading: torchvision.datasets.CIFAR10 (download=True if missing), with a fallback to
data/cifar10_arrays.npz (same arrays, used when the download is not possible).

`get_dataloader(cid, batch_size)` is what FedLab's SerialClientTrainer expects.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedlab.utils.dataset.partition import CIFAR10Partitioner

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def _to_tensor(x_uint8_nhwc, y):
    x = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc)).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - MEAN) / STD
    return x, torch.as_tensor(np.asarray(y), dtype=torch.long)


def _load_cifar10(root):
    npz = os.path.join(root, "cifar10_arrays.npz")
    try:
        import torchvision
        tr = torchvision.datasets.CIFAR10(os.path.join(root, "cifar10"), train=True, download=True)
        te = torchvision.datasets.CIFAR10(os.path.join(root, "cifar10"), train=False, download=True)
        return _to_tensor(tr.data, tr.targets), _to_tensor(te.data, te.targets)
    except Exception as e:  # no network, proxy, etc.
        if not os.path.exists(npz):
            raise RuntimeError(f"CIFAR-10 download failed ({e}) and {npz} not found. "
                               "Run `python download_data.py` on a machine with internet, or copy data/cifar10_arrays.npz.")
        d = np.load(npz)
        return _to_tensor(d["x_train"], d["y_train"]), _to_tensor(d["x_test"], d["y_test"])


class FedCIFAR10:
    def __init__(self, root, num_clients, partition="iid", dir_alpha=0.3, shards_per_client=2, seed=0, verbose=False):
        self.root, self.num_clients, self.partition, self.seed = root, num_clients, partition, seed
        (self.train_x, self.train_y), (self.test_x, self.test_y) = _load_cifar10(root)

        targets = self.train_y.numpy()
        if partition == "iid":
            p = CIFAR10Partitioner(targets, num_clients, balance=True, partition="iid", seed=seed, verbose=verbose)
        elif partition == "noniid-shards":
            p = CIFAR10Partitioner(targets, num_clients, balance=None, partition="shards",
                                   num_shards=shards_per_client * num_clients, seed=seed, verbose=verbose)
        elif partition == "noniid-dir":
            p = CIFAR10Partitioner(targets, num_clients, balance=None, partition="dirichlet",
                                   dir_alpha=dir_alpha, seed=seed, verbose=verbose)
        else:
            raise ValueError(f"unknown partition {partition}")
        self.client_dict = {cid: np.asarray(idx) for cid, idx in p.client_dict.items()}

    # ---- API used by FedLab trainers ----
    def get_dataset(self, cid, type="train"):
        idx = torch.as_tensor(self.client_dict[cid], dtype=torch.long)
        return TensorDataset(self.train_x[idx], self.train_y[idx])

    def get_dataloader(self, cid, batch_size=64, type="train"):
        return DataLoader(self.get_dataset(cid), batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size=1000):
        return DataLoader(TensorDataset(self.test_x, self.test_y), batch_size=batch_size)

    # ---- statistics ----
    def client_sizes(self):
        return [len(self.client_dict[c]) for c in range(self.num_clients)]

    def class_histogram(self):
        hist = np.zeros((self.num_clients, 10), dtype=int)
        y = self.train_y.numpy()
        for c in range(self.num_clients):
            hist[c] = np.bincount(y[self.client_dict[c]], minlength=10)
        return hist

    def client_classes(self, cid):
        return sorted(set(self.train_y[self.client_dict[cid]].tolist()))
