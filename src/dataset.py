import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.constants import IMG_SIZE, NUM_WORKERS, BATCH_SIZE


def _stratified_subset(dataset, n_per_class: int) -> Subset:
    """
    Restituisce un Subset bilanciato: esattamente n_per_class immagini
    per ogni classe, prese dalle prime occorrenze trovate nel dataset.

    Args:
        dataset     : ImageFolder (o qualsiasi dataset con .targets)
        n_per_class : numero di campioni per classe

    Returns:
        Subset con n_per_class * num_classes indici
    """
    # Raggruppa gli indici per classe
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_indices[label].append(idx)

    selected = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        if len(indices) < n_per_class:
            raise ValueError(
                f"Classe {label} ha solo {len(indices)} campioni, "
                f"ma ne richiedi {n_per_class}."
            )
        selected.extend(indices[:n_per_class])

    return Subset(dataset, selected)


def get_dataloaders(
    data_dir:    str  = None,
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
    collate_fn        = None,
):
    """
    Prepara i DataLoader per ImageNet-100.

    Ritorna tre loader:
      - train_loader : tutto il training set meno i campioni del test
      - val_loader   : validation set completo
      - test_loader  : subset bilanciato estratto dal training,
                       stessa dimensione del validation set

    Il test set viene estratto PRIMA di costruire il train loader,
    così le due partizioni sono disgiunte.
    """
    if data_dir is None:
        from src.constants import DATA_DIR
        data_dir = DATA_DIR

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Val e test non hanno augmentation - solo resize + crop deterministico
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # --- Carica i dataset grezzi ---
    # Per il test usiamo eval_transform (no augmentation)
    # Per estrarre gli indici usiamo lo stesso dataset con eval_transform
    train_dataset_aug  = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_transform)
    train_dataset_eval = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=eval_transform)
    val_dataset        = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),   transform=eval_transform)

    # --- Calcola n_per_class in modo da matchare il val set ---
    num_classes  = len(val_dataset.classes)
    n_per_class  = len(val_dataset) // num_classes

    # --- Costruisci il test subset (da train, con eval_transform) ---
    test_subset = _stratified_subset(train_dataset_eval, n_per_class)
    test_indices_set = set(test_subset.indices)

    # --- Train subset: tutti gli indici di train NON nel test ---
    all_train_indices = list(range(len(train_dataset_aug)))
    train_indices     = [i for i in all_train_indices if i not in test_indices_set]
    train_subset      = Subset(train_dataset_aug, train_indices)

    # --- DataLoader ---
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader