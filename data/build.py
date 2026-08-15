import torch
from data.sampler import build_subset_samplers


def get_dataset(dataset_cfg):
    if dataset_cfg.name == 'dexed':
        from data.dexeddataset import DexedDataset
        full_dataset = DexedDataset(**dataset_cfg)
    elif dataset_cfg.name == 'surge':
        from data.surgedataset import SurgeDataset
        full_dataset = SurgeDataset(**dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_cfg.name}")
        
    return full_dataset


def get_split_dataloaders(train_cfg, full_dataset, random_seed=0):
    subset_samplers = build_subset_samplers(
        full_dataset,
        k_fold=train_cfg.current_k_fold,
        k_folds_count=train_cfg.k_folds,
        test_holdout_proportion=train_cfg.test_holdout_proportion,
        random_seed=random_seed,
    )
    dataloaders = dict()
    sub_datasets_lengths = dict()

    for k, sampler in subset_samplers.items():
        drop_last = True
        dataloaders[k] = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=train_cfg.minibatch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=train_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        sub_datasets_lengths[k] = len(sampler.indices)

        print(
            f"[data/build.py] Dataset '{k}' contains",
            f"{sub_datasets_lengths[k]}/{len(full_dataset)} samples",
            f"({100.0 * sub_datasets_lengths[k]/len(full_dataset):.1f}%)",
        )
            
    return dataloaders
