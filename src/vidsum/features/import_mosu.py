"""Convert the audio/text streams of the MoSu subset h5 into the unified per-modality format.

    python -m vidsum.features.import_mosu

The MoSu h5 (TripleSumm format) stores all modalities per video: `features` (visual, read in place as
modality "visual_mosu"), `features_audio` and `features_text` (768-d, 1 per second). This writes
data/features/mosu/{audio_mosu,text_mosu}.h5 with a `mask`:

A stream's "placeholder" is a single vector repeated across many videos (e.g. the embedding of an empty
transcript where nobody speaks). Steps equal to it get mask = 0, so the model and the gating agent can
tell "modality absent" from real content.
"""
import argparse
from collections import Counter

import h5py
import numpy as np
from tqdm import tqdm

from ..paths import dataset_info, feature_path
from .io import open_feature_file, write_video

STREAMS = {"audio_mosu": "features_audio", "text_mosu": "features_text"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min_video_share", type=float, default=0.25,
                        help="A repeated vector counts as placeholder if it occurs in at least this share of videos")
    return parser.parse_args()


def find_placeholder(src, name, min_video_share):
    """Most frequent exact row of stream `name`, if it occurs in >= min_video_share of the videos."""
    videos_with = Counter()
    for key in src:
        rows = np.ascontiguousarray(src[f"{key}/{name}"][()], dtype=np.float32)
        videos_with.update({r.tobytes() for r in rows})
    row, count = videos_with.most_common(1)[0]
    if count < min_video_share * len(src):
        return None, count
    return np.frombuffer(row, dtype=np.float32), count


def main():
    args = parse_args()
    with h5py.File(dataset_info("mosu")["official_h5"], "r") as src:
        keys = sorted(src)
        for modality, name in STREAMS.items():
            placeholder, n_videos = find_placeholder(src, name, args.min_video_share)
            hdf = open_feature_file(feature_path("mosu", modality), {
                "extractor": f"MoSu / TripleSumm precomputed `{name}` (768-d, 1 fps)",
                "alignment": "identical to official picks (copied)",
                "placeholder_masked": placeholder is not None,
            })
            try:
                masked_share = []
                for key in tqdm(keys, desc=modality):
                    feats = np.asarray(src[f"{key}/{name}"][()], dtype=np.float32)
                    if placeholder is None:
                        mask = np.ones(len(feats), dtype=np.float32)
                    else:
                        mask = (~np.all(feats == placeholder, axis=1)).astype(np.float32)
                    masked_share.append(1 - mask.mean())
                    write_video(hdf, key, feats, mask, source="mosu_subset.h5")
            finally:
                hdf.close()
            status = (f"placeholder found in {n_videos}/{len(keys)} videos, masks {np.mean(masked_share):.1%} of steps"
                      if placeholder is not None else f"no placeholder (most repeated vector in {n_videos} videos)")
            print(f"{modality}: {status} -> {feature_path('mosu', modality)}")


if __name__ == "__main__":
    main()
