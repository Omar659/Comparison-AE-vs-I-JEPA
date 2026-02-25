import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(
    encoder:        nn.Module,
    target_encoder: nn.Module,
    momentum:       float,
) -> None:
    """
    Aggiorna i pesi del target encoder come EMA dell'encoder.

    Args:
        encoder        : context encoder (trainable) - sorgente dei nuovi pesi
        target_encoder : target encoder  (frozen)   - aggiornato in-place
        momentum       : coefficiente EMA in [0, 1]
                         0.996 -> target si aggiorna del 0.4% ad ogni step
                         1.0   -> target completamente frozen (nessun aggiornamento)
    """
    for param_enc, param_tgt in zip(encoder.parameters(), target_encoder.parameters()):
        param_tgt.data.mul_(momentum).add_((1.0 - momentum) * param_enc.data)


def get_momentum(
    step:       int,
    total_steps: int,
    start:      float,
    end:        float,
) -> float:
    """
    Schedule lineare del momentum EMA da start a end (stile I-JEPA / V-JEPA).
    Se start == end restituisce il valore fisso (stile V-JEPA 2).

    Args:
        step        : step corrente (0-indexed)
        total_steps : numero totale di step di training
        start       : momentum iniziale (es. 0.996)
        end         : momentum finale   (es. 1.0)

    Returns:
        momentum : float in [start, end]

    Esempio:
        momentum = get_momentum(step, total_steps, EMA_MOMENTUM_START, EMA_MOMENTUM_END)
        update_ema(model.encoder, model.target_encoder, momentum)
    """
    return start + (end - start) * step / total_steps