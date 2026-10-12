"""Multimodal dataset over the official TVSum/SumMe splits.

Every modality is read from a file in the unified feature format (see features/io.py): one group per
official key (`video_k`) holding `features` with shape [n_steps, D], aligned to the official `picks`.
The dataset's `official_modality` (paths.py: "googlenet" for TVSum/SumMe, "visual_mosu" for MoSu) reads
the official h5's own `features`, i.e. CA-SUM's input.
"""
import json

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..paths import dataset_info, feature_path

def load_split(dataset, split_index):
    with open(dataset_info(dataset)["splits"], encoding="utf-8") as handle:
        split = json.load(handle)[int(split_index)]
    return list(split["train_keys"]), list(split["test_keys"])


def _modality_file(dataset, modality):
    if modality == dataset_info(dataset)["official_modality"]:
        return dataset_info(dataset)["official_h5"]
    path = feature_path(dataset, modality)
    if not path.exists():
        raise FileNotFoundError(f"Missing features for modality {modality!r}: {path}")
    return path


class VideoSumDataset(Dataset):
    """Items are dicts: {"key": str, "features": {modality: FloatTensor[T, D]}, "masks": {modality: [T]}}.

    A per-step mask (1 = modality present) is read from the optional `mask` dataset; it defaults to ones.
    """

    def __init__(self, dataset, keys, modalities):
        self.dataset = dataset.lower()
        self.keys = list(keys)
        self.modalities = list(modalities)
        if not self.modalities:
            raise ValueError("At least one modality is required")

        with h5py.File(dataset_info(self.dataset)["official_h5"], "r") as hdf:
            n_steps = {k: int(np.asarray(hdf[f"{k}/n_steps"])) for k in self.keys}

        self.items = [{"key": k, "features": {}, "masks": {}} for k in self.keys]
        for modality in self.modalities:
            with h5py.File(_modality_file(self.dataset, modality), "r") as hdf:
                for item in self.items:
                    key = item["key"]
                    if key not in hdf:
                        raise KeyError(f"{key} missing from {modality} features")
                    feats = np.asarray(hdf[f"{key}/features"], dtype=np.float32)
                    if feats.shape[0] != n_steps[key]:
                        raise ValueError(
                            f"{modality}/{key}: {feats.shape[0]} steps, official n_steps is {n_steps[key]}. "
                            "Re-extract with features/align.py instead of resizing."
                        )
                    mask = np.asarray(hdf[f"{key}/mask"], dtype=np.float32) if "mask" in hdf[key] \
                        else np.ones(feats.shape[0], dtype=np.float32)
                    item["features"][modality] = torch.from_numpy(feats)
                    item["masks"][modality] = torch.from_numpy(mask)

    @property
    def dims(self):
        return {m: int(self.items[0]["features"][m].shape[1]) for m in self.modalities}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def build_loaders(dataset, split_index, modalities):
    train_keys, test_keys = load_split(dataset, split_index)
    train_ds = VideoSumDataset(dataset, train_keys, modalities)
    test_ds = VideoSumDataset(dataset, test_keys, modalities)
    # Videos have different lengths, so one video per step; batching is done by gradient accumulation.
    # Shuffling draws from the global torch RNG (seeded in train.py), matching CA-SUM's run order.
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=None, shuffle=False)
    return train_loader, test_loader
