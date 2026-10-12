"""Unified feature file format, shared by every modality.

    data/features/<dataset>/<modality>.h5
      attrs: extractor, dim, alignment, ... (whatever the extractor records)
      /video_k/features   float32 [n_steps, D]   one row per official step (aligned to `picks`)
      /video_k/mask       float32 [n_steps]      optional; 1 = modality present at this step
      /video_k attrs:     source, fps, ...
"""
import h5py
import numpy as np


def open_feature_file(path, file_attrs):
    """Open for appending, so interrupted extraction runs resume where they stopped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hdf = h5py.File(path, "a")
    for name, value in file_attrs.items():
        hdf.attrs[name] = value
    return hdf


def write_video(hdf, key, features, mask=None, **attrs):
    if key in hdf:
        del hdf[key]
    group = hdf.create_group(key)
    group.create_dataset("features", data=np.asarray(features, dtype=np.float32), compression="gzip")
    if mask is not None:
        group.create_dataset("mask", data=np.asarray(mask, dtype=np.float32))
    for name, value in attrs.items():
        group.attrs[name] = value
