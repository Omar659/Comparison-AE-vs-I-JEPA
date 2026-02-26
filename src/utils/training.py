import os
import json
import math

import torch


# =============================================================================
# SCHEDULE
# =============================================================================

def get_lr(
    epoch:         int,
    warmup_epochs: int,
    epochs:        int,
    base_lr:       float,
    min_lr:        float,
) -> float:
    """
    Warmup lineare per i primi warmup_epochs, poi cosine decay fino a min_lr.

    Args:
        epoch         : epoca corrente (0-indexed)
        warmup_epochs : numero di epoche di warmup
        epochs        : numero totale di epoche
        base_lr       : LR massimo raggiunto dopo il warmup
        min_lr        : LR minimo a fine cosine decay

    Returns:
        lr : float
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def get_weight_decay(
    epoch:  int,
    epochs: int,
    start:  float,
    end:    float,
) -> float:
    """
    Schedule lineare del weight decay da start a end.

    Args:
        epoch  : epoca corrente (0-indexed)
        epochs : numero totale di epoche
        start  : weight decay iniziale
        end    : weight decay finale

    Returns:
        wd : float
    """
    return start + (end - start) * epoch / max(1, epochs - 1)


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    path:      str,
    epoch:     int,
    states:    dict[str, dict],
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    metrics:   dict,
) -> None:
    """
    Salva un checkpoint.

    Args:
        path      : path completo del file .pt
        epoch     : epoca corrente
        states    : dizionario {nome: state_dict} dei moduli da salvare
                    es. {'encoder': enc.state_dict(), 'decoder': dec.state_dict()}
        optimizer : optimizer da salvare
        scaler    : GradScaler da salvare
        metrics   : dict con le metriche dell'epoca (per riferimento futuro)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Recupera il wandb run_id se disponibile
    try:
        import wandb as _wandb
        wandb_run_id = _wandb.run.id if _wandb.run is not None else None
    except Exception:
        wandb_run_id = None

    torch.save({
        'epoch':         epoch,
        'states':        states,
        'optimizer':     optimizer.state_dict(),
        'scaler':        scaler.state_dict(),
        'metrics':       metrics,
        'wandb_run_id':  wandb_run_id,
    }, path)
    print(f"  [ckpt] Salvato: {path}")


def load_checkpoint(
    path:      str,
    device:    torch.device,
) -> dict:
    """
    Carica un checkpoint.

    Args:
        path   : path del file .pt
        device : device su cui caricare

    Returns:
        dict con chiavi:
          'epoch'        : int
          'states'       : dict {nome: state_dict}
          'optimizer'    : state_dict dell'optimizer
          'scaler'       : state_dict dello scaler
          'metrics'      : dict delle metriche salvate
          'wandb_run_id' : str o None
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"  [ckpt] Caricato: {path} (epoca {ckpt['epoch']})")
    return ckpt


# =============================================================================
# JSON RESULTS
# =============================================================================

def save_results(results: list[dict] | dict, path: str) -> None:
    """
    Salva i risultati in un file JSON.

    Accetta sia una lista (risultati per epoca) che un dict (summary).

    Args:
        results : lista di dict (una per epoca) o dict (summary finale)
        path    : path completo del file .json
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def get_kl_weight(
    epoch:         int,
    warmup_epochs: int,
    kld_weight:    float,
) -> float:
    """
    KL annealing lineare: parte da 0 e raggiunge kld_weight a fine warmup.
    Evita il posterior collapse nelle prime epoche dove la recon loss domina.

    Args:
        epoch         : epoca corrente (0-indexed)
        warmup_epochs : epoche per raggiungere il peso pieno
        kld_weight    : peso target della KL

    Returns:
        peso KL corrente
    """
    if warmup_epochs <= 0:
        return kld_weight
    return kld_weight * min(1.0, epoch / warmup_epochs)