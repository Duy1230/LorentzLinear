"""Unified data loader for node classification benchmarks.

Supports: disease_nc, cora, citeseer, airport.
Returns a standardised dict compatible with the training loop.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

DATA_ROOT = Path(__file__).parent


def _sparse_npz_to_dense(path: str) -> np.ndarray:
    d = np.load(path)
    M = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
    return np.asarray(M.todense(), dtype=np.float32)


def _train_val_test_split(n: int, train_frac=0.6, val_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


def load_disease_nc() -> dict:
    root = DATA_ROOT / "disease_nc"
    feats = _sparse_npz_to_dense(str(root / "disease_nc.feats.npz"))
    labels = np.load(str(root / "disease_nc.labels.npy"), allow_pickle=True).astype(int)
    n = feats.shape[0]
    train_mask, val_mask, test_mask = _train_val_test_split(n)
    return {
        "name": "disease_nc",
        "features": torch.tensor(feats),
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "n_classes": int(labels.max()) + 1,
        "n_nodes": n,
    }


def load_cora() -> dict:
    root = DATA_ROOT / "cora"
    feats = _sparse_npz_to_dense(str(root / "cora.feats.npz"))
    labels = np.load(str(root / "cora.labels.npy"), allow_pickle=True).astype(int)
    n = feats.shape[0]
    train_mask, val_mask, test_mask = _train_val_test_split(n)
    return {
        "name": "cora",
        "features": torch.tensor(feats),
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "n_classes": int(labels.max()) + 1,
        "n_nodes": n,
    }


def load_citeseer() -> dict:
    root = DATA_ROOT / "citeseer"
    feats = _sparse_npz_to_dense(str(root / "citeseer.feats.npz"))
    labels = np.load(str(root / "citeseer.labels.npy"), allow_pickle=True).astype(int)
    n = feats.shape[0]
    train_mask, val_mask, test_mask = _train_val_test_split(n)
    return {
        "name": "citeseer",
        "features": torch.tensor(feats),
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "n_classes": int(labels.max()) + 1,
        "n_nodes": n,
    }


def load_airport() -> dict:
    root = DATA_ROOT / "airport"
    with open(str(root / "airport.p"), "rb") as f:
        G = pickle.load(f)
    with open(str(root / "airport_alldata.p"), "rb") as f:
        df = pickle.load(f)

    node_ids = list(G.nodes())
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    feats = np.array([G.nodes[nid]["feat"] for nid in node_ids], dtype=np.float32)

    gdps = np.array([
        df.iloc[nid - 1]["gdp"] if (nid - 1) < len(df) else np.nan
        for nid in node_ids
    ])
    valid = ~np.isnan(gdps)
    quartiles = np.nanquantile(gdps[valid], [0.25, 0.5, 0.75])
    labels_all = np.digitize(gdps, quartiles).astype(int)
    labels_all[~valid] = 0

    train_mask, val_mask, test_mask = _train_val_test_split(n)
    train_mask[~valid] = False
    val_mask[~valid] = False
    test_mask[~valid] = False

    return {
        "name": "airport",
        "features": torch.tensor(feats),
        "labels": torch.tensor(labels_all, dtype=torch.long),
        "train_mask": torch.tensor(train_mask),
        "val_mask": torch.tensor(val_mask),
        "test_mask": torch.tensor(test_mask),
        "n_classes": 4,
        "n_nodes": n,
    }


LOADERS = {
    "disease_nc": load_disease_nc,
    "cora": load_cora,
    "citeseer": load_citeseer,
    "airport": load_airport,
}


def load_dataset(name: str) -> dict:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADERS.keys())}")
    return LOADERS[name]()


if __name__ == "__main__":
    for name in LOADERS:
        data = load_dataset(name)
        print(f"{name}: N={data['n_nodes']}, d_feat={data['features'].shape[1]}, "
              f"classes={data['n_classes']}, "
              f"train/val/test={data['train_mask'].sum()}/{data['val_mask'].sum()}/{data['test_mask'].sum()}")
