# SynthRL Stage 1 training bundle

This repo is a minimal, self-contained training bundle that trains
SynthRL on pre-rendered Dexed spectrograms using parameter loss.

## Requirements

- Linux x86-64/iOS
- Conda (Miniconda or Anaconda)
- An NVIDIA GPU with a driver compatible with CUDA 11.7, pretty sure that latest CUDA 13.3 is backcompatible with 11.7 though
- Enough disk space for the extracted bundle and Conda environment ~4Gb

## Setup and run

### Dataset layout

Expects:

```text
datasets/dexed/
├── params.pt
└── spectrogram/
```
so just copy the contents of the unzipped file data/dataset into skripsie-training/datasets/dexed.

### Then
From repo root:

```bash
./setup.sh
./run.sh
```
This should work, lmk if something fails.



The first command creates the `synthrl-stage1` Conda environment. The second
starts the configured 200-epoch training run. Outputs and checkpoints are saved
under `run/stage1/seed2/`.

To do a one-epoch hardware benchmark first:

```bash
./run.sh train.end_epoch=1
```

To override configuration values, append Hydra arguments. For example:

```bash
./run.sh train.minibatch_size=16 train.num_workers=4 train.save_period=2
```

## Resume from a checkpoint

Checkpoints are written every five epochs. To resume at epoch 50, ensure
`run/stage1/seed2/checkpoints/00050.tar` exists, then run:

```bash
./run.sh train.start_epoch=50
```

Keep the `run/` folder when moving between machines or notebook sessions.



Parameter metadata is stored in `synth/dexed_presets.sqlite`.

## Kaggle or another managed GPU image

Managed notebook images often already provide a newer CUDA-enabled PyTorch.
Creating this pinned environment is the most reproducible route, but first
confirm that the platform has enough writable disk space for both the dataset
and environment. Run `./run.sh train.end_epoch=1` before committing to the full
run.

## Troubleshooting

- CUDA out of memory: lower `train.minibatch_size` from 32 to 16 or 8.
- Data-loader errors: set `train.num_workers=0`.
- Existing environment: remove it with `conda env remove -n synthrl-stage1`,
  then rerun `./setup.sh`.
