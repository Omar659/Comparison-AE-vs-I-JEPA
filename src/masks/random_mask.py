import torch

from src.constants import PATCH_SIZE, IMG_SIZE, VAE_MASK_RATIO


class RandomMaskGenerator:
    """
    Maschera random stile MAE: seleziona casualmente una frazione
    di patch da nascondere, senza vincoli di blocco o aspect ratio.

    Args:
        mask_ratio  : frazione di patch da mascherare (default 0.75, come MAE)
        img_size    : dimensione immagine
        patch_size  : dimensione patch
    """

    def __init__(
        self,
        mask_ratio:  float = VAE_MASK_RATIO,
        img_size:    int   = IMG_SIZE,
        patch_size:  int   = PATCH_SIZE,
    ):
        self.mask_ratio  = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.num_masked  = int(self.num_patches * mask_ratio)
        self.num_visible = self.num_patches - self.num_masked

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Genera maschere random per un batch.

        Returns:
            visible_ids : (B, N_visible) indici delle patch visibili
            mask_ids    : (B, N_masked)  indici delle patch mascherate
        """
        visible_ids = []
        mask_ids    = []

        for _ in range(batch_size):
            perm = torch.randperm(self.num_patches)
            visible_ids.append(perm[:self.num_visible])
            mask_ids.append(perm[self.num_visible:])

        visible_ids = torch.stack(visible_ids)   # (B, N_visible)
        mask_ids    = torch.stack(mask_ids)      # (B, N_masked)

        # Ordina per rendere il gather deterministico
        visible_ids, _ = visible_ids.sort(dim=1)
        mask_ids,    _ = mask_ids.sort(dim=1)

        return visible_ids, mask_ids