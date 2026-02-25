import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vit import VisionTransformerEncoder
from src.models.predictor import Predictor
from src.constants import EMBED_DIM


class IJEPA(nn.Module):
    """
    Image Joint-Embedding Predictive Architecture.

    Args:
        encoder      : istanza di VisionTransformerEncoder (context encoder)
        predictor    : istanza di Predictor
    
    Il target_encoder viene creato internamente come deepcopy dell'encoder
    con tutti i parametri frozen.
    """

    def __init__(
        self,
        encoder:   VisionTransformerEncoder,
        predictor: Predictor,
    ):
        super().__init__()

        self.encoder   = encoder
        self.predictor = predictor

        # Target encoder: copia profonda dell'encoder, parametri frozen
        # I pesi vengono aggiornati esternamente via EMA (src/utils/ema.py)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        x:           torch.Tensor,
        visible_ids: torch.Tensor,
        mask_ids:    torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass completo: context encoder -> predictor -> loss L1.

        Args:
            x           : (B, C, H, W) immagini
            visible_ids : (B, N_vis)   indici patch visibili
            mask_ids    : (B, N_mask)  indici patch mascherate

        Returns:
            loss : scalar - L1 medio sulle patch mascherate
            pred : (B, N_mask, embed_dim) - predizioni del predictor
                   (utile per logging o analisi)
        """
        # --- Target: EMA encoder vede l'immagine intera ---
        # stop-gradient: i gradienti non fluiscono nel target_encoder
        with torch.no_grad():
            all_tokens = self.target_encoder(x, mask_indices=None)  # (B, N, D)

            # Estrai solo i token corrispondenti alle patch mascherate
            # Questi sono i TARGET che il predictor deve imparare a predire
            idx = mask_ids.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
            target = torch.gather(all_tokens, dim=1, index=idx)     # (B, N_mask, D)

        # --- Context encoder: vede solo le patch visibili ---
        visible_tokens = self.encoder(x, mask_indices=visible_ids)  # (B, N_vis, D)

        # --- Predictor: predice le rappresentazioni delle patch mascherate ---
        pred = self.predictor(visible_tokens, visible_ids, mask_ids)  # (B, N_mask, D)

        # --- Loss L1 solo sulle patch mascherate ---
        loss = F.l1_loss(pred, target)

        return loss, pred

    def trainable_parameters(self):
        """
        Restituisce solo i parametri trainabili: encoder + predictor.
        Il target_encoder è escluso (frozen, aggiornato via EMA).

        Uso tipico:
            optimizer = AdamW(model.trainable_parameters(), lr=...)
        """
        return (
            list(self.encoder.parameters()) +
            list(self.predictor.parameters())
        )

    @torch.no_grad()
    def initialize_target_encoder(self):
        """
        Copia i pesi dell'encoder nel target_encoder.
        Da chiamare esplicitamente se i pesi dell'encoder cambiano
        prima dell'inizio del training (es. dopo un resume da checkpoint).
        """
        for param_enc, param_tgt in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_tgt.data.copy_(param_enc.data)