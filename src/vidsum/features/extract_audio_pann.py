"""PANNs (Cnn14, AudioSet) audio embeddings, one 2048-d vector per official step.

    python -m vidsum.features.extract_audio_pann --dataset tvsum [--keys video_1 video_2] [--device cuda]

Each step gets the embedding of a `--window`-second audio window centered on the step's timestamp.
Full 2048-d output is stored; any reduction is learned by the model (no fixed random projection).
mask = 0 where the window is silent (RMS below --silence_rms) or the video has no audio track.
Resumable: keys already in the output file are skipped unless --overwrite.
"""
import argparse

import av
import numpy as np
import torch
from tqdm import tqdm

from ..paths import feature_path
from .align import official_keys, step_times, video_path
from .io import open_feature_file, write_video

SAMPLE_RATE = 32000  # PANNs requirement


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["tvsum", "summe"])
    parser.add_argument("--keys", nargs="*", help="Subset of official keys (default: all)")
    parser.add_argument("--window", type=float, default=1.0, help="Audio window length in seconds")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--silence_rms", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_audio(path):
    """Decode the first audio stream to mono float32 at SAMPLE_RATE with PyAV (bundled FFmpeg).

    librosa's fallback decoder on macOS cannot read PCM-in-mp4 (all of SumMe), so PyAV is used everywhere.
    Returns an empty array if there is no audio stream; keeps the decoded part of a corrupt stream.
    """
    with av.open(str(path)) as container:
        if not container.streams.audio:
            print(f"  no audio stream in {path.name}")
            return np.zeros(0, dtype=np.float32)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        try:
            for frame in container.decode(audio=0):
                chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        except av.error.FFmpegError as exc:
            print(f"  audio of {path.name} is corrupt after {sum(map(len, chunks)) / SAMPLE_RATE:.1f}s: {exc}")
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def centered_windows(audio, times, window):
    half = int(window * SAMPLE_RATE / 2)
    starts = (times * SAMPLE_RATE).astype(np.int64)  # index into padded == centered in the original
    # Zero-pad both ends; the tail also covers steps past the decoded audio (short or missing tracks).
    tail = max(half, int(starts.max(initial=0)) + 2 * half - (len(audio) + half))
    padded = np.pad(audio, (half, tail))
    return np.stack([padded[s:s + 2 * half] for s in starts])


def main():
    args = parse_args()
    from panns_inference import AudioTagging

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = AudioTagging(checkpoint_path=None, device=device)

    out_path = feature_path(args.dataset, "audio_pann")
    hdf = open_feature_file(out_path, {
        "extractor": "PANNs Cnn14 AudioSet (panns_inference), 2048-d embedding",
        "sample_rate": SAMPLE_RATE,
        "window_sec": args.window,
        "alignment": "window centered on official picks / fps",
    })
    try:
        keys = args.keys or official_keys(args.dataset)
        for key in tqdm(keys, desc=f"PANN {args.dataset}"):
            if key in hdf and not args.overwrite:
                continue
            path = video_path(args.dataset, key)
            times, fps = step_times(args.dataset, key, path)
            audio = load_audio(path)
            windows = centered_windows(audio, times, args.window)

            feats = []
            with torch.no_grad():
                for i in range(0, len(windows), args.batch):
                    _, emb = model.inference(windows[i:i + args.batch])
                    feats.append(emb)
            feats = np.concatenate(feats).astype(np.float32)
            mask = (np.sqrt((windows ** 2).mean(axis=1)) > args.silence_rms).astype(np.float32)
            write_video(hdf, key, feats, mask, source=path.name, fps=fps, has_audio=bool(len(audio)))
    finally:
        hdf.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
