"""Torch dataset in superblock space + collate + splits."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .basis import NSLOT, to_superblocks
from .data_prep import FlatQH9


def all_pairs(n):
    src, dst = np.where(~np.eye(n, dtype=bool))
    return src.astype(np.int64), dst.astype(np.int64)


class SuperblockDataset(Dataset):
    def __init__(self, flat: FlatQH9, indices):
        self.flat, self.indices = flat, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        z, pos, H = self.flat.raw(i)
        n = len(z)
        T, mask = to_superblocks(H.astype(np.float32), z)
        ar = np.arange(n)
        src, dst = all_pairs(n)
        return dict(
            z=torch.from_numpy(z), pos=torch.from_numpy(pos.astype(np.float32)),
            src=torch.from_numpy(src), dst=torch.from_numpy(dst),
            onsite=torch.from_numpy(T[ar, ar]), onsite_mask=torch.from_numpy(mask[ar, ar]),
            pair=torch.from_numpy(T[src, dst]), pair_mask=torch.from_numpy(mask[src, dst]),
            idx=i)


def collate(batch):
    off, out = 0, {k: [] for k in ("z", "pos", "src", "dst", "onsite", "onsite_mask", "pair", "pair_mask", "batch")}
    for b, item in enumerate(batch):
        n = len(item["z"])
        for k in ("z", "pos", "onsite", "onsite_mask", "pair", "pair_mask"):
            out[k].append(item[k])
        out["src"].append(item["src"] + off)
        out["dst"].append(item["dst"] + off)
        out["batch"].append(torch.full((n,), b, dtype=torch.long))
        off += n
    res = {k: torch.cat(v) for k, v in out.items()}
    res["idx"] = [it["idx"] for it in batch]
    return res


def make_splits(flat: FlatQH9, kind="id", seed=0, subset=None, frac=(0.8, 0.1, 0.1)):
    """'id'  : random split.  'size': test = largest molecules (extrapolation), val random from the rest."""
    rng = np.random.default_rng(seed)
    n = len(flat)
    idx = rng.permutation(n)[: subset or n]
    if kind == "id":
        a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        return idx[:a], idx[a:b], idx[b:]
    sizes = np.array([flat.n_atoms(int(i)) for i in idx])
    order = idx[np.argsort(sizes, kind="stable")]
    n_test = int(frac[2] * len(order))
    test, rest = order[len(order) - n_test:], rng.permutation(order[: len(order) - n_test])
    n_val = int(frac[1] * len(idx))
    return rest[n_val:], rest[:n_val], test


def estimate_scale(flat: FlatQH9, indices, n=200):
    vals = [np.abs(flat.raw(int(i))[2]).ravel() for i in list(indices)[:n]]
    return float(np.sqrt(np.mean(np.concatenate(vals) ** 2)))
