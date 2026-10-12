"""Central registry of dataset locations. Everything else asks this module for paths.

Defaults are <repo>/data and <repo>/experiments. Override with environment variables, e.g. on Kaggle:
    VIDSUM_DATA=/kaggle/input/vidsum-data          (read-only is fine for training)
    VIDSUM_EXPERIMENTS=/kaggle/working/experiments
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("VIDSUM_DATA", ROOT / "data")).expanduser().resolve()
OFFICIAL = DATA / "official"
FEATURES = DATA / "features"
RAW = DATA / "raw"
EXPERIMENTS = Path(os.environ.get("VIDSUM_EXPERIMENTS", ROOT / "experiments")).expanduser().resolve()

# official_modality: name under which the official h5's own `features` are exposed to configs.
# rank_gt: ground truth for Kendall/Spearman: "tsv" (per-annotator TVSum scores), "gtscore", or None.
DATASETS = {
    "tvsum": {
        "official_h5": OFFICIAL / "tvsum" / "eccv16_dataset_tvsum_google_pool5.h5",
        "splits": OFFICIAL / "splits" / "tvsum_splits.json",
        "anno_tsv": OFFICIAL / "tvsum" / "ydata-anno.tsv",
        "videos": RAW / "tvsum" / "videos",
        "official_modality": "googlenet",
        "eval_method": "avg",  # F1 averaged over the 20 annotators
        "rank_gt": "tsv",
    },
    "summe": {
        "official_h5": OFFICIAL / "summe" / "eccv16_dataset_summe_google_pool5.h5",
        "splits": OFFICIAL / "splits" / "summe_splits.json",
        "anno_tsv": None,
        "videos": RAW / "summe" / "videos",
        "official_modality": "googlenet",
        "eval_method": "max",  # F1 of the best-matching annotator
        "rank_gt": None,
    },
    # 200-video subsample of MoSu (TripleSumm, ICLR 2026): 1 fps, ground truth = YouTube "most replayed".
    "mosu": {
        "official_h5": OFFICIAL / "mosu" / "mosu_subset.h5",
        "splits": OFFICIAL / "splits" / "mosu_splits.json",
        "anno_tsv": None,
        "videos": RAW / "mosu" / "videos",  # not available; features come precomputed
        "official_modality": "visual_mosu",
        "eval_method": "avg",  # a single (crowd) summary, so avg == max
        "rank_gt": "gtscore",
    },
}


def dataset_info(name):
    name = name.lower()
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def feature_path(dataset, modality):
    """Unified feature files live at data/features/<dataset>/<modality>.h5."""
    return FEATURES / dataset.lower() / f"{modality}.h5"
