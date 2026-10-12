# data/ (gitignored except this file)

```
official/                      what every experiment and metric reads
  tvsum/eccv16_dataset_tvsum_google_pool5.h5   GoogLeNet pool5 features + ground truth (Zhang et al., ECCV16)
  tvsum/ydata-anno.tsv         per-frame importance scores of the 20 annotators (used for tau/rho)
  tvsum/ydata-info.tsv         category, title, URL and length per video (titles: candidate text input)
  summe/eccv16_dataset_summe_google_pool5.h5
  mosu/mosu_subset.h5          200-video MoSu subsample (TripleSumm format, 1 fps): ground truth +
                               768-d `features` (visual), `features_audio`, `features_text`
  splits/{tvsum,summe,mosu}_splits.json        5 random 80/20 splits
raw/                           source material for feature extraction
  tvsum/videos/<youtube_id>.mp4   50 videos, from ydata-tvsum50-video.zip
  tvsum/README.txt                official TVSum README
  summe/videos/<name>.mp4         25 videos, from SumMe.zip (webm duplicates not extracted)
  summe/GT/<name>.mat             per-frame user selections (user_score [n_frames, n_users]), gt_score, FPS
  summe/README.txt                official SumMe README
features/<dataset>/<modality>.h5  unified, step-aligned format (src/vidsum/features/io.py)
  tvsum/audio_pann.h5, summe/audio_pann.h5     PANN Cnn14, 2048-d
  mosu/audio_mosu.h5, mosu/text_mosu.h5        MoSu's own streams in unified format (vidsum.features.import_mosu);
                                               text steps equal to the "no transcript" placeholder have mask = 0
features/legacy/               symlinks to older files, NOT in the unified format; do not train on them as-is
  tvsum_resnet_pann_rp512.h5   ResNet50 + PANN, 1 fps, random-projected to 512, keyed by YouTube id
  tvsum_whisper.h5, summe_whisper.h5  512-d, n_steps-aligned, origin unverified (likely Whisper encoder)
  summe_clip.h5                1024-d CLIP, n_steps-aligned, origin unverified
```

Sources: `implementations/data/datasets 2/` (SumMe.zip, tvsum50_ver_1_1.tar). Its two eccv16 h5 files are
byte-identical to the ones here. The extracted TVSum videos are byte-identical to the copy in
`PANN_getaudio/datasets/tvsum50`.

## Known data issues

- **SumMe `Uncut_Evening_Flight` (video_21) is truncated.** Both the mp4 and the webm are corrupt inside
  `SumMe.zip`: 7,909 of 9,672 frames decode and audio stops at 266 s of 322 s. The official GoogLeNet
  features and ground truth are complete, so training and F1 are unaffected. Audio steps past 266 s are
  zero with `mask = 0`. Re-download this video before extracting any new visual or text features for it.
- **SumMe `Scuba` and `St Maarten Landing` have no audio track.** Their audio features are zero, with `mask = 0`.
- The other 23 SumMe videos carry PCM audio in mp4, which macOS's built-in decoder can't read; decoding uses PyAV.

## MoSu notes

- Ground truth is YouTube "most replayed" (one crowd signal): `user_summary` has a single row, and
  tau/rho are computed against `gtscore`. mAP uses 5-step shots, i.e. 5 s at 1 fps, as in TripleSumm.
- The text stream's placeholder vector covers 27.7% of steps (in 188/200 videos); audio has none.
- Results on this subset are not comparable to TripleSumm's published full-MoSu numbers (supervised,
  ~50k videos). Compare methods with each other on this subset.
- Reference points, split 0: random scores give F1 51.3, tau ~0, mAP50 54.9.

## Key mapping

TVSum `video_k` = the k-th distinct id in `ydata-anno.tsv`. Verified: n_frames matches the annotation
length for all 50 videos. SumMe stores `video_name` in the h5, which is the file name in `raw/summe/videos`.
