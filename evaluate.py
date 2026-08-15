import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.build import get_dataset, get_split_dataloaders
from logs.logger import get_model_checkpoint
from model.loss import (CategoricalParamsAccuracy, PresetProcessor,
                        QuantizedNumericalParamsLoss)
from model.network import SynthRL
from utils.audio import AudioRenderer, Spectrogram_Processor


def evaluate(cfg: DictConfig):
    torch.set_num_threads(cfg.cpu_threads)
    torch.set_num_interop_threads(min(4, cfg.cpu_threads))
    logs_root_dir = Path(to_absolute_path(cfg.logs_root_dir))
    log_dir = logs_root_dir / cfg.run_name
    train_cfg = OmegaConf.load(log_dir / "config.yaml")
    train_cfg.train.minibatch_size = cfg.minibatch_size
    train_cfg.train.num_workers = cfg.loader_workers

    eval_path = log_dir / cfg.dataset.name / f"{cfg.ckpt_epoch:03d}epoch"
    audio_path = eval_path / "audio"
    audio_path.mkdir(parents=True, exist_ok=True)
    eval_pkl_path = eval_path / "eval.pickle"

    synth_cfg = OmegaConf.load(f"config/dataset/{cfg.synth_name}.yaml")
    synth_dataset = get_dataset(synth_cfg)
    eval_dataset = get_dataset(cfg.dataset)
    dataloader = get_split_dataloaders(train_cfg.train, eval_dataset)[cfg.split]
    preset_idx_helper = synth_dataset.preset_indexes_helper

    device = torch.device(cfg.device)
    ckpt = get_model_checkpoint(log_dir, cfg.ckpt_epoch, device)
    model = SynthRL(preset_idx_helper, **train_cfg.model).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    param_num_criterion = QuantizedNumericalParamsLoss(
        preset_idx_helper, nn.L1Loss(), reduce=False
    )
    param_cat_criterion = CategoricalParamsAccuracy(
        preset_idx_helper, reduce=False, percentage_output=True
    )

    sample_rate = train_cfg.dataset.sample_rate
    preset_processor = PresetProcessor(synth_dataset, preset_idx_helper, device)
    audio_renderer = AudioRenderer(
        synth_dataset, sample_rate, num_workers=1, device=device
    )
    spec_processor = Spectrogram_Processor(
        train_cfg.dataset.n_fft, train_cfg.dataset.fft_hop, sample_rate
    ).to(device)

    eval_metrics = []
    assert cfg.minibatch_size == 1

    try:
        from pyvirtualdisplay import Display
        display = Display() if not os.environ.get("DISPLAY") else nullcontext()
    except Exception:
        display = nullcontext()

    with display, torch.inference_mode():
        for sample in tqdm(dataloader):
            x_wav, x_in, v_in, preset_uid, _ = sample
            x_in = x_in.to(device)
            v_in = v_in.to(device)
            preset_uid = preset_uid.item()

            v_out = model(x_in)
            full_preset_out, _ = preset_processor(v_out, deterministic=True)
            inferred_wav = audio_renderer.single_process_render(full_preset_out)

            x_wav = x_wav.to(device)

            inferred_wav = inferred_wav / torch.abs(inferred_wav + 1e-5).max(1)[0]
            x_wav = x_wav / torch.abs(x_wav + 1e-5).max(1)[0]
            sc, log_mae, mfcc13_mae, mfcc40_mae = (
                spec_processor.calculate_metrics(x_wav, inferred_wav)
            )

            sf.write(audio_path / f"{preset_uid}_gt.wav", x_wav.cpu().numpy()[0], sample_rate)
            sf.write(audio_path / f"{preset_uid}.wav", inferred_wav.cpu().numpy()[0], sample_rate)

            accuracies = param_cat_criterion(v_out, v_in)
            acc_value = np.asarray(list(accuracies.values())).mean()
            _, mae_value = param_num_criterion(v_out, v_in)
            eval_metrics.append({
                "preset_UID": preset_uid,
                "spec_sc": sc.item(),
                "spec_mae": log_mae.item(),
                "mfcc13_mae": mfcc13_mae.item(),
                "mfcc40_mae": mfcc40_mae.item(),
                "num_mae": mae_value.item(),
                "cat_acc": acc_value,
            })

    eval_df = pd.DataFrame(eval_metrics)
    eval_df = eval_df.groupby("preset_UID", as_index=False).mean(numeric_only=True)
    eval_df.to_pickle(eval_pkl_path)
    print(f"Finished evaluation: {eval_pkl_path}")


@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
