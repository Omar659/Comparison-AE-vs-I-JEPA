import torch
import torch.nn as nn

from src.constants import (
    EMBED_DIM,
    PRED_EMBED_DIM, PRED_DEPTH, PRED_NUM_HEADS,
    MLP_RATIO, DROP, ATTN_DROP,
    IMG_SIZE, PATCH_SIZE,
)
from src.models.vit import Block, build_2d_sincos_pos_embed


class Predictor(nn.Module):
    """
    Predictor narrow ViT per image-JEPA (ispirato a V-JEPA 2).

    Args:
        encoder_dim    : dimensione embedding dell'encoder (default da constants)
        pred_embed_dim : dimensione interna del predictor, più stretta (default 384)
        depth          : numero di blocchi transformer (default 6)
        num_heads      : teste di attenzione (default 6)
        mlp_ratio      : rapporto MLP hidden / pred_embed_dim
        drop           : dropout
        attn_drop      : dropout sull'attention
        num_patches    : numero totale di patch dell'immagine (default da IMG_SIZE/PATCH_SIZE)
    """

    def __init__(
        self,
        encoder_dim:    int   = EMBED_DIM,
        pred_embed_dim: int   = PRED_EMBED_DIM,
        depth:          int   = PRED_DEPTH,
        num_heads:      int   = PRED_NUM_HEADS,
        mlp_ratio:      float = MLP_RATIO,
        drop:           float = DROP,
        attn_drop:      float = ATTN_DROP,
        num_patches:    int   = (IMG_SIZE // PATCH_SIZE) ** 2,
    ):
        super().__init__()

        self.encoder_dim    = encoder_dim
        self.pred_embed_dim = pred_embed_dim
        self.num_patches    = num_patches

        # -----------------------------------------------------------------
        # 1. Proiezione input: encoder_dim -> pred_embed_dim
        #    I token in uscita dall'encoder hanno dimensione encoder_dim (768),
        #    ma il predictor lavora in uno spazio più stretto (384).
        # -----------------------------------------------------------------
        self.input_proj = nn.Linear(encoder_dim, pred_embed_dim)

        # -----------------------------------------------------------------
        # 2. Positional embedding (sinusoidale, non trainable)
        #    Stesso pos embed del ViT encoder: ogni posizione ha un vettore
        #    univoco. Lo usiamo sia per i token visibili che per i mask token.
        # -----------------------------------------------------------------
        grid_size  = int(num_patches ** 0.5)
        pos_embed  = build_2d_sincos_pos_embed(pred_embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed)  # (1, N, pred_dim)

        # -----------------------------------------------------------------
        # 3. Mask token learnable
        #    Un unico vettore (1, 1, pred_dim) che viene replicato per ogni
        #    patch mascherata. È il "segnaposto" che il predictor deve riempire.
        #    Inizializzato a zero - imparerà durante il training.
        # -----------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))

        # -----------------------------------------------------------------
        # 4. Blocchi Transformer narrow
        # -----------------------------------------------------------------
        self.blocks = nn.ModuleList([
            Block(pred_embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_embed_dim)

        # -----------------------------------------------------------------
        # 5. Proiezione output: pred_embed_dim -> encoder_dim
        #    Le predizioni devono avere la stessa dimensione dei target
        #    prodotti dall'EMA encoder (encoder_dim = 768).
        # -----------------------------------------------------------------
        self.output_proj = nn.Linear(pred_embed_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Il mask token parte da zero (come in V-JEPA / MAE)
        nn.init.zeros_(self.mask_token)

    def forward(
        self,
        context_tokens: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens  : (B, N_vis, encoder_dim)
                              Output dell'encoder sui token visibili.

            visible_indices : (B, N_vis)
                              Indici delle patch visibili nell'immagine originale.
                              Serve per estrarre il pos embed corretto.

            masked_indices  : (B, N_masked)
                              Indici delle patch mascherate da predire.
                              Serve per dare al mask token la posizione giusta.

        Returns:
            predictions : (B, N_masked, encoder_dim)
                          Predizioni nello spazio dell'encoder,
                          da confrontare con i target dell'EMA encoder.
        """
        B          = context_tokens.size(0)
        N_vis      = context_tokens.size(1)
        N_masked   = masked_indices.size(1)

        # -----------------------------------------------------------------
        # Step 1 - Proietta i token visibili in pred_dim
        # -----------------------------------------------------------------
        x = self.input_proj(context_tokens)          # (B, N_vis, pred_dim)

        # -----------------------------------------------------------------
        # Step 2 - Aggiungi pos embed ai token visibili
        #   Il pos embed completo ha shape (1, N_total, pred_dim).
        #   Estraiamo solo le posizioni visibili con gather.
        # -----------------------------------------------------------------
        pos_vis = self.pos_embed.expand(B, -1, -1)   # (B, N_total, pred_dim)
        idx_vis = visible_indices.unsqueeze(-1).expand(-1, -1, self.pred_embed_dim)
        x = x + torch.gather(pos_vis, dim=1, index=idx_vis)

        # -----------------------------------------------------------------
        # Step 3 - Costruisci i mask tokens con pos embed delle posizioni mascherate
        #   mask_token  : (1, 1, pred_dim) -> replicato -> (B, N_masked, pred_dim)
        #   pos_masked  : pos embed estratto per le posizioni mascherate
        # -----------------------------------------------------------------
        mask_tokens = self.mask_token.expand(B, N_masked, -1).clone()

        idx_msk     = masked_indices.unsqueeze(-1).expand(-1, -1, self.pred_embed_dim)
        pos_masked  = torch.gather(pos_vis, dim=1, index=idx_msk)

        mask_tokens = mask_tokens + pos_masked       # (B, N_masked, pred_dim)

        # -----------------------------------------------------------------
        # Step 4 - Concatena: [token visibili | mask tokens]
        #   Il transformer può fare self-attention sull'intera sequenza,
        #   permettendo ai mask token di "guardare" i token visibili.
        # -----------------------------------------------------------------
        tokens = torch.cat([x, mask_tokens], dim=1)  # (B, N_vis+N_masked, pred_dim)

        # -----------------------------------------------------------------
        # Step 5 - Transformer blocks
        # -----------------------------------------------------------------
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # -----------------------------------------------------------------
        # Step 6 - Estrai solo le posizioni mascherate e proietta in encoder_dim
        #   I token visibili sono i primi N_vis, i mask token sono gli ultimi N_masked.
        # -----------------------------------------------------------------
        pred = tokens[:, N_vis:, :]                  # (B, N_masked, pred_dim)
        pred = self.output_proj(pred)                # (B, N_masked, encoder_dim)

        return pred