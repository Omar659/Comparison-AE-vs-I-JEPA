import torch
import torch.nn as nn

from src.constants import (
    IMG_SIZE, PATCH_SIZE,
    EMBED_DIM, ENCODER_DEPTH, NUM_HEADS,
    MLP_RATIO, DROP, ATTN_DROP, CHANNELS,
)


# =============================================================================
# 1. PATCH EMBEDDING
# =============================================================================

class PatchEmbed(nn.Module):
    """
    Divide l'immagine in patch non-overlapping e le proietta in embed_dim.

    Input  : (B, C, H, W)
    Output : (B, N, embed_dim)   dove N = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) deve essere divisibile per patch_size ({patch_size})"

        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv 2D con kernel = stride = patch_size ≡ split + linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/p, W/p) -> (B, N, embed_dim)
        x = self.proj(x)                   # (B, D, H/p, W/p)
        x = x.flatten(2)                   # (B, D, N)
        x = x.transpose(1, 2)             # (B, N, D)
        return x


# =============================================================================
# 2. POSITIONAL EMBEDDING 2D (sinusoidale)
# =============================================================================

def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Costruisce un positional embedding sinusoidale 2D.

    Args:
        embed_dim : dimensione dell'embedding (deve essere pari)
        grid_size : numero di patch per lato (es. 224/16 = 14)

    Returns:
        pos_embed : (1, grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim deve essere pari per il pos embedding 2D"

    half_dim = embed_dim // 2

    # Frequenze per le due metà
    omega = torch.arange(half_dim // 2, dtype=torch.float32) / (half_dim // 2)
    omega = 1.0 / (10000 ** omega)                              # (half_dim/2,)

    # Griglia 2D
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')  # (G, G)

    # Encoding per asse H e asse W separatamente
    def sincos(coords: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        coords = coords.flatten().unsqueeze(1)  # (N, 1)
        out    = coords * omega.unsqueeze(0)    # (N, half_dim/2)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # (N, half_dim)

    emb_h = sincos(grid_h, omega)   # (N, half_dim)
    emb_w = sincos(grid_w, omega)   # (N, half_dim)

    pos_embed = torch.cat([emb_h, emb_w], dim=-1).unsqueeze(0)  # (1, N, embed_dim)
    return pos_embed


# =============================================================================
# 3. ATTENTION
# =============================================================================

class Attention(nn.Module):
    """Multi-head self-attention con proiezioni fuse (qkv)."""

    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv        = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.attn_drop  = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                 # ciascuno: (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        return x


# =============================================================================
# 4. TRANSFORMER BLOCK
# =============================================================================

class Block(nn.Module):
    """
    Singolo blocco Transformer: LayerNorm -> Attention -> residual
                                 LayerNorm -> MLP      -> residual
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = Attention(embed_dim, num_heads, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 5. VIT ENCODER - la classe principale
# =============================================================================

class VisionTransformerEncoder(nn.Module):
    """
    ViT Encoder riusabile per JEPA, AE e VAE. Default parameters for ViT-base

    Comportamento:
      - Se `mask_indices` è None  -> processa TUTTE le patch (EMA encoder, AE)
      - Se `mask_indices` è fornito -> processa solo le patch VISIBILI (context encoder)

    Il positional embedding viene aggiunto PRIMA del drop dei token,
    garantendo che ogni token porti la sua informazione posizionale anche
    dopo il masking.

    Args:
        img_size    : dimensione dell'immagine quadrata
        patch_size  : dimensione di ogni patch
        in_chans    : canali input
        embed_dim   : dimensione embedding
        depth       : numero di blocchi transformer
        num_heads   : teste di attenzione
        mlp_ratio   : rapporto hidden dim MLP / embed_dim
        drop        : dropout generale
        attn_drop   : dropout sull'attention map
    """

    def __init__(
        self,
        img_size:   int   = IMG_SIZE,
        patch_size: int   = PATCH_SIZE,
        in_chans:   int   = CHANNELS,
        embed_dim:  int   = EMBED_DIM,
        depth:      int   = ENCODER_DEPTH,
        num_heads:  int   = NUM_HEADS,
        mlp_ratio:  float = MLP_RATIO,
        drop:       float = DROP,
        attn_drop:  float = ATTN_DROP,
    ):
        super().__init__()

        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.grid_size   = img_size // patch_size    # es. 14 per 224/16
        self.num_patches = self.grid_size ** 2       # es. 196

        # --- Blocchi principali ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Positional embedding: (1, N, D) - NON trainable (sinusoidale)
        pos_embed = build_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Inizializzazione pesi stile ViT (Xavier uniform per linear, zero bias)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                mask_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x            : immagine (B, C, H, W)
            mask_indices : (B, N_visible) - indici delle patch da TENERE.
                           None -> tutte le patch (EMA encoder / AE).

        Returns:
            tokens : (B, N_out, embed_dim)
                     N_out = N_visible se mask_indices fornito, altrimenti N totale
        """
        # 1. Patch embedding: (B, N, D)
        tokens = self.patch_embed(x)

        # 2. Aggiungi positional embedding PRIMA del drop
        tokens = tokens + self.pos_embed

        # 3. Drop dei token mascherati (se richiesto)
        if mask_indices is not None:
            # mask_indices: (B, N_visible) - gather lungo dim 1
            idx = mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            tokens = torch.gather(tokens, dim=1, index=idx)

        # 4. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # 5. Layer norm finale
        tokens = self.norm(tokens)

        return tokens