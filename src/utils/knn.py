import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.constants import KNN_K, KNN_TEMPERATURE


@torch.no_grad()
def extract_features(
    encoder:    nn.Module,
    loader:     DataLoader,
    device:     torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estrae le rappresentazioni globali dell'encoder per tutto il dataset.

    Args:
        encoder : VisionTransformerEncoder (o qualsiasi encoder)
        loader  : DataLoader del dataset
        device  : device su cui eseguire

    Returns:
        features : (N, D) - rappresentazioni L2-normalizzate
        labels   : (N,)   - label corrispondenti
    """
    encoder.eval()
    all_features = []
    all_labels   = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)

        # Forward completo: nessuna maschera -> tutti i token
        tokens = encoder(imgs, mask_indices=None)   # (B, N, D)

        # Average pooling -> rappresentazione globale per immagine
        feats = tokens.mean(dim=1)                  # (B, D)

        # L2 normalizzazione -> distanza coseno = prodotto scalare
        feats = F.normalize(feats, dim=-1)

        all_features.append(feats.cpu())
        all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)   # (N, D)
    labels   = torch.cat(all_labels,   dim=0)   # (N,)

    return features, labels


@torch.no_grad()
def knn_accuracy(
    encoder:      nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    k:            int   = KNN_K,
    temperature:  float = KNN_TEMPERATURE,
) -> float:
    """
    Calcola la k-NN accuracy usando le rappresentazioni dell'encoder.

    Usa il metodo weighted k-NN di DINO/I-JEPA: invece del semplice
    voto a maggioranza, i vicini contribuiscono con peso proporzionale
    alla loro similarità coseno (scalata dalla temperatura).
    Questo è più robusto del voto uniforme.

    Args:
        encoder      : encoder da valutare (messo in eval automaticamente)
        train_loader : DataLoader del train set (feature bank)
        test_loader  : DataLoader del test set
        device       : device
        k            : numero di vicini (default 20, standard in letteratura)
        temperature  : scala la distribuzione dei pesi (default 0.07)

    Returns:
        accuracy : float in [0, 1]
    """
    # --- Estrai feature bank (train) ---
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats,  test_labels  = extract_features(encoder, test_loader,  device)

    train_feats  = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats   = test_feats.to(device)
    test_labels  = test_labels.to(device)

    num_classes = int(train_labels.max().item()) + 1
    correct     = 0
    total       = test_feats.shape[0]

    # Processa il test set a chunk per non saturare la VRAM
    chunk_size = 256
    for start in range(0, total, chunk_size):
        end   = min(start + chunk_size, total)
        query = test_feats[start:end]              # (chunk, D)

        # Similarità coseno tra query e feature bank
        # query: (chunk, D), train_feats: (N_train, D)
        sim = torch.mm(query, train_feats.T)       # (chunk, N_train)

        # Scala per temperatura e prendi i top-k per ogni query
        sim = sim / temperature
        top_sim, top_idx = sim.topk(k, dim=-1)    # (chunk, k)

        # Pesi softmax sulle similarità
        weights = top_sim.softmax(dim=-1)          # (chunk, k)

        # Label dei k vicini
        top_labels = train_labels[top_idx]         # (chunk, k)

        # Voto pesato: accumula i pesi per classe
        votes = torch.zeros(end - start, num_classes, device=device)
        votes.scatter_add_(
            dim=1,
            index=top_labels,
            src=weights,
        )

        # Predizione = classe con più voti pesati
        preds = votes.argmax(dim=1)                # (chunk,)
        correct += (preds == test_labels[start:end]).sum().item()

    return correct / total