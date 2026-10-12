"""Map official keys to raw video files and official steps to timestamps.

Every extractor computes one feature per official step, centered on that step's frame (picks / fps),
so all modalities share the exact time axis of the GoogLeNet features and ground truth. No resizing.
"""
import csv
import warnings

import av
import h5py
import numpy as np

from ..paths import dataset_info

VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv")


def official_keys(dataset):
    with h5py.File(dataset_info(dataset)["official_h5"], "r") as hdf:
        return sorted(hdf.keys(), key=lambda k: int(k.split("_")[1]))


def _tvsum_ids(tsv_path):
    ids = []
    with open(tsv_path, encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row[0] not in ids:
                ids.append(row[0])
    return ids


def video_path(dataset, key):
    """Raw video file of an official key. TVSum keys follow TSV order; SumMe stores the name in the h5."""
    info = dataset_info(dataset)
    if dataset == "tvsum":
        stem = _tvsum_ids(info["anno_tsv"])[int(key.split("_")[1]) - 1]
    else:
        with h5py.File(info["official_h5"], "r") as hdf:
            stem = hdf[f"{key}/video_name"][()].decode()
    for ext in VIDEO_EXTS:
        path = info["videos"] / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No video file for {dataset}/{key} ({stem}) in {info['videos']}")


def step_times(dataset, key, path):
    """Seconds of each official step in the raw video, plus the video fps."""
    with h5py.File(dataset_info(dataset)["official_h5"], "r") as hdf:
        picks = np.asarray(hdf[f"{key}/picks"], dtype=np.float64)
        n_frames = int(np.asarray(hdf[f"{key}/n_frames"]))
    # PyAV, not OpenCV: both bundle FFmpeg and loading the two in one process can crash on macOS.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        file_frames = int(stream.frames)
    if fps <= 0:
        raise RuntimeError(f"Cannot read fps of {path}")
    if abs(file_frames - n_frames) > 0.01 * n_frames:
        warnings.warn(f"{key}: video has {file_frames} frames, official n_frames={n_frames}; check the file")
    return picks / fps, fps
