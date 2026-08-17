# SynthRL three-stage training bundle

This is a self-contained training bundle for the three-stage SynthRL pipeline.
The current priority is to continue Stage 2 and Stage 3 from the already-trained
Stage 1 seed-2 checkpoint. Multi-seed Stage 1 replication remains set up for
later paper replication.

## Requirements

- Linux x86-64
- Conda (Miniconda or Anaconda)
- An NVIDIA GPU and a driver compatible with the pinned CUDA 11.7 PyTorch build
- At least 24 GB for both datasets, the Conda environment, and initial outputs
- Git LFS (used for the trained checkpoint and bundled synthesizer binaries)

## Priority 1: set up the environment and data

From the repository root:

```bash
./setup.sh
```

The datasets must have this layout:

```text
datasets/
├── dexed/
│   ├── params.pt
│   └── spectrogram/
└── surge/              # installed automatically before Stage 3
    ├── wav/
    └── spectrogram/
```

Copy the extracted Dexed dataset into `datasets/dexed/`. The complete Surge
dataset is a versioned GitHub Release asset; `./setup_data.sh` downloads it,
verifies its SHA-256 checksum, and extracts it into `datasets/surge/`. Stage 3
calls this script automatically and does not require `surgepy`.

Stage 1 can read pre-rendered data without RenderMan. RL fine-tuning renders
inferred Dexed presets, so the Linux x86-64 runtime binaries are included via
Git LFS at:

```text
third_party/RenderMan/Builds/LinuxMakefile/build/librenderman.so
third_party/Dexed/Builds/Linux/build/Dexed.so
```

Their licenses and pinned corresponding-source links are in
`third_party/README.md`. If these builds are incompatible with the target
machine, build replacements and set both paths before fine-tuning:

```bash
export SYNTHRL_RENDERMAN_BUILD_DIR=/absolute/path/to/RenderMan/Builds/LinuxMakefile/build
export SYNTHRL_DEXED_PLUGIN_PATH=/absolute/path/to/Dexed.so
```

## Priority 2: run Stage 2 from trained seed 2

The tracked checkpoint must exist at
`run/stage1/seed2/checkpoints/00199.tar`. Then run:

```bash
./run_stage2.sh
```

Stage 2 loads Stage 1 epoch 199, uses seed 2, and writes to
`run/stage2/seed2/`. A one-epoch smoke test that uses a separate output is:

```bash
./run_stage2.sh run_name=smoke/stage2-seed2 train.end_epoch=200 loss.synth_render_workers=2
```

Hydra overrides can be appended to any command, for example
`train.save_period=10` or `loss.synth_render_workers=8`.

## Priority 3: run Stage 3 from Stage 2 seed 2

After Stage 2 finishes, its checkpoint must exist at
`run/stage2/seed2/checkpoints/00399.tar`. Then run:

```bash
./run_stage3.sh
```

Stage 3 loads that checkpoint, keeps seed 2, trains on the Surge dataset, and
writes to `run/stage3/seed2/`. On first use it downloads the approximately
1.5 GB compressed dataset release. A one-epoch smoke test is:

```bash
./run_stage3.sh run_name=smoke/stage3-seed2 train.end_epoch=400 loss.synth_render_workers=2
```

Keep the complete `run/` directory when moving between machines because each
stage reads the preceding stage's saved configuration and checkpoint.

## Later priority: replicate Stage 1 across seeds

The existing seed multiplication is retained for the later full paper
replication. Train seeds 1 and 3 consecutively with:

```bash
./run_stage1_seeds.sh
```

They write to `run/stage1/seed1/` and `run/stage1/seed3/`. Seed 3 starts
only after seed 1 succeeds. Overrides apply to both runs:

```bash
./run_stage1_seeds.sh train.minibatch_size=16 train.num_workers=4
```

To run either seed separately:

```bash
./run.sh seed=1 run_name=stage1/seed1
./run.sh seed=3 run_name=stage1/seed3
```

## Stage 1 and checkpoint maintenance

The default Stage 1 command remains `./run.sh`; it trains seed 2 for 200
epochs and writes to `run/stage1/seed2/`. To resume from epoch 50:

```bash
./run.sh train.start_epoch=50
```

To transfer the final checkpoint with Git LFS:

```bash
git lfs install
git lfs track "*.tar"
git add .gitattributes run/stage1/seed2/checkpoints/00199.tar
git commit -m "Add trained Stage 1 model"
git push
```

## Publishing the data asset

The downloader expects a release tagged `data-v1` with an asset named
`surge-dataset-v1.tar.gz`. The prepared archive has this SHA-256 checksum:

```text
fb6d0e75513a9969975fda6543aa1baee5a74e59bf90d597683dcc6076d1098d
```

After installing and authenticating GitHub CLI, publish it from the machine
where the archive was prepared:

```bash
gh release create data-v1 /tmp/surge-dataset-v1.tar.gz \
  --title "SynthRL Surge dataset v1" \
  --notes "Pre-rendered Surge waveforms and spectrograms for Stage 3 training."
```

## Troubleshooting

- CUDA out of memory: lower `train.minibatch_size` to 16 or 8.
- Data-loader errors: set `train.num_workers=0`.
- Render errors: reduce `loss.synth_render_workers`, verify both renderer
  paths, and check `DISPLAY`/Xvfb availability.
- Existing environment: remove it with
  `conda env remove -n synthrl-stage1`, then rerun `./setup.sh`.
