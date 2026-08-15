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

## Run the remaining Stage 1 seeds

Seeds 1 and 3 can be trained consecutively with one command:

```bash
./run_stage1_seeds.sh
```

The runs use independent RNG seeds and save to `run/stage1/seed1/` and
`run/stage1/seed3/`, respectively. Seed 3 starts only after seed 1 completes
successfully. Any Hydra overrides are applied to both runs:

```bash
./run_stage1_seeds.sh train.minibatch_size=16 train.num_workers=4
```

To run either seed on its own:

```bash
./run.sh seed=1 run_name=stage1/seed1
./run.sh seed=3 run_name=stage1/seed3
```

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

## Transfer the trained model

After the default training run finishes, push the final checkpoint with Git
LFS (install Git LFS first if it is not already available):

```bash
git lfs install
git lfs track "*.tar"
git add .gitattributes run/stage1/seed2/checkpoints/00199.tar
git commit -m "Add trained Stage 1 model"
git push
```

The recipient can then run `git pull` with Git LFS installed. The trained model
will be at `run/stage1/seed2/checkpoints/00199.tar`.



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
