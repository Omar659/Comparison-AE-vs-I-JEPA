import torch
import torch.nn as nn

from src.constants import (
    IMG_SIZE, PATCH_SIZE, CHANNELS,
    EMBED_DIM, DECODER_EMBED_DIM, DECODER_DEPTH, DECODER_NUM_HEADS,
    MLP_RATIO, DROP, ATTN_DROP
)
from src.models.vit import Block, build_2d_sincos_pos_embed

class VisionTransformerDecoder(nn.Module):
    """
    Decoder Transformer stile MAE.
    Proietta l'output dell'encoder nello spazio del decoder, 
    aggiunge i mask token per le patch mancanti (se presenti), 
    applica i blocchi di attenzione e infine proietta nello spazio dei pixel.
    """
    def __init__(
        self,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        in_chans: int = CHANNELS,
        embed_dim: int = EMBED_DIM,
        decoder_dim: int = DECODER_EMBED_DIM,
        depth: int = DECODER_DEPTH,
        num_heads: int = DECODER_NUM_HEADS,
        mlp_ratio: float = MLP_RATIO,
        drop: float = DROP,
        attn_drop: float = ATTN_DROP,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # 1. Proiezione dall'encoder/latente allo spazio del decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        
        # 2. Mask Token (rappresenta le patch da ricostruire se stiamo mascherando)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # 3. Positional Embedding (1, N, D)
        pos_embed = build_2d_sincos_pos_embed(decoder_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed)
        
        # 4. Blocchi Transformer
        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(decoder_dim)
        
        # 5. Proiezione finale verso i pixel (patch_size * patch_size * in_chans)
        self.pred = nn.Linear(decoder_dim, patch_size**2 * in_chans)

        self._init_weights()

    def _init_weights(self):
        # Inizializzazione mask token stile MAE (distribuzione normale)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N_vis, embed_dim) - Token visibili o latenti
        ids_restore: (B, N_patches) - Indici originali per ricostruire l'ordine spaziale.
                     Se None, assume che x contenga già tutte le patch (B, N_patches, embed_dim).
        """
        # Proietta nella dimensione del decoder
        x = self.decoder_embed(x)
        B, L, D = x.shape
        
        if ids_restore is not None:
            # Ricostruisce la sequenza completa inserendo i mask_token dove mancano le patch
            mask_tokens = self.mask_token.expand(B, self.num_patches - L, -1)
            # x: [B, N_vis, D] concatenato con [B, N_mask, D] -> [B, N_total, D]
            x_ = torch.cat([x, mask_tokens], dim=1)
            # Rimescola i token per rimetterli nella posizione originale dell'immagine
            ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, D)
            x = torch.gather(x_, dim=1, index=ids_restore_expanded)
            
        # Aggiunge il positional embedding
        x = x + self.pos_embed

        # Applica i blocchi Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Proietta nello spazio dei pixel: (B, N_patches, patch_size**2 * 3)
        x = self.pred(x)
        
        return x