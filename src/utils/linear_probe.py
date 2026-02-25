import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.utils.knn import extract_features
from src.constants import LINEAR_PROBE_EPOCHS, LINEAR_PROBE_LR


def linear_probe(
    encoder:      nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    num_classes:  int,
    epochs:       int   = LINEAR_PROBE_EPOCHS,
    lr:           float = LINEAR_PROBE_LR,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    """
    Allena un classificatore lineare frozen sulle rappresentazioni dell'encoder
    e valuta l'accuracy sul test set.

    Args:
        encoder      : encoder da valutare (sempre frozen durante la probe)
        train_loader : DataLoader del train set
        test_loader  : DataLoader del test set
        device       : device
        num_classes  : numero di classi del dataset
        epochs       : epoche di training del classificatore (default 20)
        lr           : learning rate del classificatore (default 1e-3)
        weight_decay : weight decay del classificatore (default 0.0)

    Returns:
        dict con chiavi:
          'top1' : top-1 accuracy in [0, 1]
          'top5' : top-5 accuracy in [0, 1]
    """
    # --- Step 1: estrai le rappresentazioni una volta sola (encoder frozen) ---
    # È più efficiente estrarre tutto e poi allenare il classificatore
    # sulle feature pre-calcolate, invece di fare un forward encoder
    # ad ogni batch del probe training
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)

    embed_dim = train_feats.shape[1]

    # --- Step 2: DataLoader sulle feature pre-calcolate ---
    train_ds     = TensorDataset(train_feats, train_labels)
    train_loader_feats = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    # --- Step 3: classificatore lineare ---
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    nn.init.normal_(classifier.weight, std=0.01)
    nn.init.zeros_(classifier.bias)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # --- Step 4: training del classificatore ---
    classifier.train()
    for epoch in range(epochs):
        for feats, labels in train_loader_feats:
            feats  = feats.to(device)
            labels = labels.to(device)

            logits = classifier(feats)
            loss   = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- Step 5: valutazione sul test set ---
    classifier.eval()
    test_feats  = test_feats.to(device)
    test_labels = test_labels.to(device)

    top1_correct = 0
    top5_correct = 0
    total        = test_feats.shape[0]

    chunk_size = 256
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end    = min(start + chunk_size, total)
            feats  = test_feats[start:end]
            labels = test_labels[start:end]

            logits = classifier(feats)              # (chunk, num_classes)

            # Top-1
            preds_top1 = logits.argmax(dim=1)
            top1_correct += (preds_top1 == labels).sum().item()

            # Top-5
            top5_preds = logits.topk(min(5, num_classes), dim=1).indices
            for i, label in enumerate(labels):
                if label in top5_preds[i]:
                    top5_correct += 1

    return {
        'top1': top1_correct / total,
        'top5': top5_correct / total,
    }