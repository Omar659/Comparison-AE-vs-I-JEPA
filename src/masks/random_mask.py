import torch

# Importiamo le costanti
from src.constants import IMG_SIZE, PATCH_SIZE, MASK_RATIO

class RandomMaskGenerator:
    # Usiamo le costanti come parametri di default!
    def __init__(self, input_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO):
        self.num_patches = (input_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.num_mask = int(self.num_patches * mask_ratio)
        self.num_vis = self.num_patches - self.num_mask

    def __call__(self, batch_size):
        # Genera rumore casuale per ogni elemento del batch
        noise = torch.rand(batch_size, self.num_patches)
        
        # L'argsort ordina gli indici mischiandoli casualmente
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Dividiamo tra visibili e mascherati
        visible_ids = ids_shuffle[:, :self.num_vis]
        mask_ids = ids_shuffle[:, self.num_vis:]
        
        return visible_ids, mask_ids