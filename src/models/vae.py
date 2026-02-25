# import torch
# import torch.nn as nn

# from src.models.vit import VisionTransformerEncoder
# from src.models.decoder import VisionTransformerDecoder
# from src.constants import EMBED_DIM, VAE_LATENT_DIM

# class VariationalMaskedAE(nn.Module):
#     """
#     Variational Masked Autoencoder (VAE).
    
#     1. Passa le patch visibili all'Encoder.
#     2. Proietta l'output in mu e logvar.
#     3. Usa il reparameterization trick per campionare z.
#     4. Ricostruisce la sequenza completa calcolando le patch mancanti e inserendo i mask_token.
#     5. Passa z al Decoder per ricostruire i pixel originali.
#     """
#     def __init__(
#         self,
#         encoder: VisionTransformerEncoder,
#         decoder: VisionTransformerDecoder,
#         embed_dim: int = EMBED_DIM,
#         latent_dim: int = VAE_LATENT_DIM
#     ):
#         super().__init__()
        
#         self.encoder = encoder
#         self.decoder = decoder
        
#         # Proiezioni per calcolare Media e Log-Varianza per ogni token
#         self.fc_mu = nn.Linear(embed_dim, latent_dim)
#         self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
#         self._init_weights()

#     def _init_weights(self):
#         # Inizializzazione standard per le proiezioni latenti
#         nn.init.xavier_uniform_(self.fc_mu.weight)
#         nn.init.zeros_(self.fc_mu.bias)
#         nn.init.xavier_uniform_(self.fc_logvar.weight)
#         nn.init.zeros_(self.fc_logvar.bias)

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """
#         Reparameterization trick: z = mu + std * epsilon
#         """
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + eps * std
#         else:
#             return mu

#     def forward(
#         self, 
#         x: torch.Tensor, 
#         visible_ids: torch.Tensor, 
#         mask_ids: torch.Tensor = None
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             x: (B, C, H, W) immagini originali
#             visible_ids: (B, N_vis) indici delle patch visibili
#             mask_ids: opzionale, NON USATO internamente per via del troncamento. 
#                       Viene calcolato il complemento esatto dinamicamente.
            
#         Returns:
#             pred_pixel_patches: (B, N_total, patch_size**2 * C) patch ricostruite
#             mu: (B, N_vis, latent_dim)
#             logvar: (B, N_vis, latent_dim)
#         """
#         # 1. Encoding delle sole patch visibili
#         # (B, N_vis, embed_dim)
#         encoded_tokens = self.encoder(x, mask_indices=visible_ids)
        
#         # 2. Bottleneck Variazionale
#         # (B, N_vis, latent_dim)
#         mu = self.fc_mu(encoded_tokens)
#         logvar = self.fc_logvar(encoded_tokens)
        
#         # (B, N_vis, latent_dim)
#         z = self.reparameterize(mu, logvar)
        
#         # 3. Costruzione di ids_restore per rimettere i token nell'ordine spaziale
#         B, N_vis = visible_ids.shape
#         N_total = self.encoder.num_patches
        
#         # Troviamo l'esatto complemento di visible_ids.
#         # Questo garantisce che N_vis + N_mask_exact sia SEMPRE uguale a N_total (es. 196)
#         mask_bool = torch.ones(B, N_total, dtype=torch.bool, device=x.device)
#         mask_bool.scatter_(1, visible_ids, False)
        
#         # Estraiamo gli indici che sono rimasti 'True' (quelli mascherati)
#         actual_mask_ids = torch.nonzero(mask_bool, as_tuple=True)[1].view(B, N_total - N_vis)
        
#         # Concatenando visible e masked otteniamo tutti i 196 indici rimescolati
#         ids_shuffle = torch.cat([visible_ids, actual_mask_ids], dim=1)
        
#         # argsort ci restituisce l'ordine esatto per ripristinare le posizioni originali!
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
        
#         # 4. Decoding
#         # Il decoder inserirà automaticamente i mask_tokens nelle posizioni mancanti
#         # e userà ids_restore per rimettere in ordine i token.
#         pred_pixel_patches = self.decoder(z, ids_restore=ids_restore)
        
#         return pred_pixel_patches, mu, logvar
        
#     def trainable_parameters(self):
#         """Restituisce tutti i parametri addestrabili per l'optimizer."""
#         return [p for p in self.parameters() if p.requires_grad]



import torch
import torch.nn as nn

from src.models.vit import VisionTransformerEncoder
from src.models.decoder import VisionTransformerDecoder
from src.constants import EMBED_DIM, VAE_LATENT_DIM


class VariationalMaskedAE(nn.Module):
    """
    Variational Masked Autoencoder con latente GLOBALE.

    Flusso:
      1. Encoder vede solo le patch visibili  -> (B, N_vis, embed_dim)
      2. Average pool sulle patch visibili    -> (B, embed_dim)
      3. fc_mu / fc_logvar                   -> mu, logvar: (B, latent_dim)
      4. Reparameterize                       -> z: (B, latent_dim)
      5. fc_z + expand                        -> (B, N_vis, latent_dim)
         (ogni posizione visibile riceve lo stesso z globale)
      6. Decoder (con ids_restore)            -> (B, N_total, P^2 * C)

    Vantaggi rispetto al per-token:
      - Un solo vettore z rappresenta l'intera immagine (come un VAE classico)
      - KL su una sola gaussiana (latent_dim-dim), non su N_vis gaussiane
      - Si puo campionare z ~ N(0,I) per generazione pura
      - Confronto piu pulito con I-JEPA (entrambi hanno repr. globale)
    """

    def __init__(
        self,
        encoder:    VisionTransformerEncoder,
        decoder:    VisionTransformerDecoder,
        embed_dim:  int = EMBED_DIM,
        latent_dim: int = VAE_LATENT_DIM,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # Bottleneck variazionale globale
        self.fc_mu     = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        # Proiezione da z globale allo spazio atteso dal decoder
        # Il decoder e istanziato con embed_dim=latent_dim, per cui
        # fc_z proietta latent_dim -> latent_dim (utile hook per futuri cambi di dim)
        self.fc_z = nn.Linear(latent_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for fc in (self.fc_mu, self.fc_logvar, self.fc_z):
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self,
        x:           torch.Tensor,
        visible_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (B, C, H, W) immagini originali
            visible_ids : (B, N_vis)   indici delle patch visibili

        Returns:
            pred  : (B, N_total, patch_size**2 * C) patch ricostruite
            mu    : (B, latent_dim)   media del latente globale
            logvar: (B, latent_dim)   log-varianza del latente globale
        """
        B, N_vis = visible_ids.shape
        N_total  = self.encoder.num_patches

        # ------------------------------------------------------------------
        # 1. Encoding: solo patch visibili -> (B, N_vis, embed_dim)
        # ------------------------------------------------------------------
        encoded = self.encoder(x, mask_indices=visible_ids)

        # ------------------------------------------------------------------
        # 2. Global average pool -> (B, embed_dim)
        # ------------------------------------------------------------------
        pooled = encoded.mean(dim=1)

        # ------------------------------------------------------------------
        # 3. Bottleneck variazionale globale
        # ------------------------------------------------------------------
        mu     = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z      = self.reparameterize(mu, logvar)   # (B, latent_dim)

        # ------------------------------------------------------------------
        # 4. Proietta z e replica su ogni posizione visibile
        #    z e globale ma il decoder si aspetta una sequenza
        # ------------------------------------------------------------------
        z_proj = self.fc_z(z)                               # (B, latent_dim)
        z_exp  = z_proj.unsqueeze(1).expand(-1, N_vis, -1) # (B, N_vis, latent_dim)

        # ------------------------------------------------------------------
        # 5. Calcola ids_restore: complemento esatto di visible_ids
        # ------------------------------------------------------------------
        mask_bool = torch.ones(B, N_total, dtype=torch.bool, device=x.device)
        mask_bool.scatter_(1, visible_ids, False)
        actual_mask_ids = torch.nonzero(mask_bool, as_tuple=True)[1].view(B, N_total - N_vis)

        ids_shuffle = torch.cat([visible_ids, actual_mask_ids], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # ------------------------------------------------------------------
        # 6. Decoding
        # ------------------------------------------------------------------
        pred = self.decoder(z_exp, ids_restore=ids_restore)   # (B, N_total, P^2*C)

        return pred, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Generazione pura: campiona z ~ N(0, I) e decodifica senza masking.

        Args:
            n      : numero di immagini da generare
            device : device

        Returns:
            pred : (n, N_total, patch_size**2 * C)
        """
        N_total   = self.decoder.num_patches
        latent_dim = self.fc_mu.out_features
        z      = torch.randn(n, latent_dim, device=device)
        z_proj = self.fc_z(z)
        z_exp  = z_proj.unsqueeze(1).expand(-1, N_total, -1)
        with torch.no_grad():
            pred = self.decoder(z_exp, ids_restore=None)
        return pred

    def trainable_parameters(self):
        """Tutti i parametri addestrabili."""
        return [p for p in self.parameters() if p.requires_grad]