# vidsum: unsupervised multimodal video summarization

Graduation project (ITU, advisor Assoc. Prof. Dr. Yusuf Yaslan). CA-SUM (concentrated attention) is
extended with audio and text features, and a reinforcement-learning agent decides per segment which
modalities to use. Training is fully unsupervised. Results are reported as F1, Kendall's τ / Spearman's ρ
and mAP on TVSum and SumMe.

## Layout

```
configs/            one YAML per experiment
data/               gitignored; see data/README.md (mostly symlinks to the older folders)
src/vidsum/
  paths.py          dataset registry (all paths come from here)
  data/dataset.py   multimodal loader over official splits; any modality subset
  features/         extractors writing the unified, step-aligned feature format (features/io.py)
  models/           ca_attention.py (CA-SUM block), ca_sum.py, fusion.py, gating_agent.py
  objectives/       losses.py (unsupervised), rewards.py (RL rewards)
  eval/             knapsack, F1, rank correlation, mAP, KTS; metrics.py is the single entry point
  train.py          one (config, sigma, split) run
  evaluate.py       label-free model selection + 5-split report
scripts/run_splits.sh   sigma x split sweep, then evaluate
scripts/pack_kaggle_data.sh  bundle data/ for upload as a Kaggle Dataset
notebooks/kaggle_train.ipynb clone + run on a Kaggle GPU
tests/              eval sanity checks, parity of the port with upstream CA-SUM
experiments/        gitignored run outputs
```

## Setup

```bash
cd vidsum
../.venv/bin/python -m pip install -r requirements.txt          # training/eval
../.venv/bin/python -m pip install -r requirements-extract.txt  # + feature extraction
../.venv/bin/python -m pytest -q
```
No install is needed to run: `scripts/run_splits.sh` puts `src/` on `PYTHONPATH`. Data and output
locations can be overridden with `VIDSUM_DATA` and `VIDSUM_EXPERIMENTS`.

## Running on Kaggle (GPU)

Code goes to GitHub and data to a Kaggle Dataset (data is gitignored).
1. `bash scripts/pack_kaggle_data.sh` → `dist/vidsum-data.zip` (~160 MB: official h5/splits/annotations + aligned features; no raw videos).
2. kaggle.com → Datasets → New Dataset → upload the zip (private).
3. Push this repo to GitHub, then import `notebooks/kaggle_train.ipynb` into a new Kaggle notebook and set its `REPO`.
4. Notebook settings: Accelerator **GPU T4**, Internet **On**, Add Input → your dataset; for a private repo add a
   `GITHUB_TOKEN` secret. Run all; for full sweeps use *Save Version → Save & Run All*.

## Usage

```bash
# Phase 1: reproduce CA-SUM (about 10 min per run on a laptop CPU, 25 runs per dataset)
bash scripts/run_splits.sh configs/casum_googlenet.yaml tvsum
bash scripts/run_splits.sh configs/casum_googlenet.yaml summe

# quick smoke test
SIGMAS="0.6" SPLITS="0" EPOCHS=5 EXP=smoke bash scripts/run_splits.sh configs/casum_googlenet.yaml tvsum

# features (resumable)
python -m vidsum.features.extract_audio_pann --dataset tvsum

# Phase 3: static fusion ablations
bash scripts/run_splits.sh configs/va_concat.yaml tvsum

# re-evaluate; --oracle also prints test-selected F1 (diagnostic only, never report it)
python -m vidsum.evaluate --exp casum_googlenet --dataset tvsum --oracle
```

## Evaluation protocol

The protocol is the same as CA-SUM's, so numbers are comparable to the paper:
- **F1**: step scores are upsampled to frames, averaged per KTS shot, and shots are picked by knapsack
  at a 15% budget. TVSum averages over annotators, SumMe takes the max.
- **τ / ρ** (TVSum only), Otani et al. 2019: correlation with each annotator's scores at the step
  positions, averaged over annotators, then over videos.
- **Model selection uses no test labels.** Per split: argmin of the training loss after epoch 20, then the
  sigma with the best label-free "improvement" score (`evaluate.py` docstring).
- Reference points (tests/test_eval.py): random scores give F1 ≈ 55–58 and τ ≈ 0 on TVSum.

## Roadmap

- [x] **Phase 0**: project structure, restored TVSum h5, unified eval (F1, τ, ρ, mAP), tests
- [ ] **Phase 1**: reproduce CA-SUM on TVSum and SumMe (target 61.4 / 51.1 F1, τ 0.160, ρ 0.210); run SegSum the same way
- [ ] **Phase 2**: features
  - [x] PANN audio extractor (step-aligned, 2048-d, PyAV decoding); TVSum + SumMe
  - [x] raw videos + official annotations for both datasets (see data/README.md for known issues)
  - [ ] re-download SumMe `Uncut_Evening_Flight` (truncated in SumMe.zip)
  - [ ] text: Whisper ASR transcripts and/or VLM frame captions, then sentence embeddings (`features/extract_text.py`)
  - [ ] verify origin of the legacy Whisper/CLIP files, or drop them
- [ ] **Phase 3**: static fusion ablations (V, A, T, V+A, V+T, V+A+T × concat / xattn / softgate)
- [ ] **Phase 4**: RL modality gating (`models/gating_agent.py` + `objectives/rewards.py`, REINFORCE, λ sweep)
- [ ] **Phase 5**: full benchmark + efficiency (params, train time < 10 min, latency, compute saved by gating)
- [ ] **Phase 6**: `infer.py`, mp4 + ratio → summary mp4 (FFmpeg), chunked attention for 2-hour videos
- [ ] **Phase 7**: report and poster
