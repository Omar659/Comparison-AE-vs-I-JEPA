import math
import torch

from src.constants import (
    IMG_SIZE, PATCH_SIZE,
    NUM_MASK_BLOCKS, MASK_BLOCK_SCALE, MASK_ASPECT_RATIO,
    MIN_VISIBLE_PATCHES,
)


class MaskGenerator:
    """
    Genera maschere multi-block per una singola immagine.

    Args:
        img_size          : dimensione immagine quadrata (default IMG_SIZE)
        patch_size        : dimensione patch (default PATCH_SIZE)
        num_blocks        : numero di blocchi da campionare e unire
        block_scale       : (min, max) frazione di area coperta da ogni blocco
        aspect_ratio      : (min, max) aspect ratio altezza/larghezza del blocco
        min_visible       : numero minimo di patch visibili garantite
    """

    def __init__(
        self,
        img_size:    int   = IMG_SIZE,
        patch_size:  int   = PATCH_SIZE,
        num_blocks:  int   = NUM_MASK_BLOCKS,
        block_scale: tuple = MASK_BLOCK_SCALE,
        aspect_ratio: tuple = MASK_ASPECT_RATIO,
        min_visible: int   = MIN_VISIBLE_PATCHES,
    ):
        self.grid_h      = img_size // patch_size   # es. 14
        self.grid_w      = img_size // patch_size   # es. 14
        self.num_patches = self.grid_h * self.grid_w  # es. 196

        self.num_blocks   = num_blocks
        self.block_scale  = block_scale
        self.aspect_ratio = aspect_ratio
        self.min_visible  = min_visible

    def _sample_block(self) -> torch.Tensor:
        """
        Campiona un singolo blocco rettangolare casuale sulla griglia di patch.

        Returns:
            mask : (grid_h, grid_w) BoolTensor - True nelle patch del blocco
        """
        # 1. Campiona l'area del blocco come frazione delle patch totali
        scale = self.block_scale[0] + torch.rand(1).item() * (
            self.block_scale[1] - self.block_scale[0]
        )
        area = int(self.num_patches * scale)   # numero patch nel blocco

        # 2. Campiona l'aspect ratio e calcola h, w
        ar = self.aspect_ratio[0] + torch.rand(1).item() * (
            self.aspect_ratio[1] - self.aspect_ratio[0]
        )
        # area = h * w,  ar = h / w  ->  h = sqrt(area * ar), w = sqrt(area / ar)
        h = int(round(math.sqrt(area * ar)))
        w = int(round(math.sqrt(area / ar)))
        
        # Clamp per stare dentro la griglia
        h = max(1, min(h, self.grid_h))
        w = max(1, min(w, self.grid_w))

        # 3. Campiona il corner in alto a sinistra del blocco
        top  = torch.randint(0, self.grid_h - h + 1, (1,)).item()
        left = torch.randint(0, self.grid_w - w + 1, (1,)).item()

        # 4. Costruisci la maschera 2D
        mask = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        mask[top:top + h, left:left + w] = True

        return mask

    def __call__(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Genera maschere per un intero batch.

        Args:
            batch_size : numero di immagini nel batch

        Returns:
            visible_ids : (B, N_vis)    indici patch visibili  - stesso N_vis per tutto il batch
            mask_ids    : (B, N_mask)   indici patch mascherate - stesso N_mask per tutto il batch

        Nota: per batch processing efficiente, tutte le immagini del batch
        hanno lo stesso N_vis (truncated al minimo trovato nel batch).
        """
        all_visible = []
        all_masked  = []
        min_visible = self.num_patches  # minimo N_vis nel batch
        min_masked  = self.num_patches  # minimo N_mask nel batch

        for _ in range(batch_size):
            vis_ids, msk_ids = self._generate_single()
            all_visible.append(vis_ids)
            all_masked.append(msk_ids)
            min_visible = min(min_visible, len(vis_ids))
            min_masked  = min(min_masked,  len(msk_ids))

        # --- Allinea le lunghezze per il batch (tronca al minimo) ---
        # Necessario perché torch.stack richiede tensori della stessa shape.
        # Visible e masked vengono troncati INDIPENDENTEMENTE al proprio minimo:
        # non è detto che N_vis + N_mask = N dopo la truncation, ma va bene
        # perché la loss si calcola solo sulle patch presenti in mask_ids.
        min_visible = max(min_visible, self.min_visible)

        visible_ids = torch.stack([v[:min_visible] for v in all_visible])  # (B, N_vis)
        mask_ids    = torch.stack([m[:min_masked]  for m in all_masked])   # (B, N_mask)
        
        return visible_ids, mask_ids

    def _generate_single(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Genera le maschere per una singola immagine.

        Strategia:
          - Campiona num_blocks blocchi e prende la loro UNIONE come maschera
          - Se troppo poche patch restano visibili, riduce progressivamente
            il numero di blocchi usati finché non si rispetta min_visible

        Returns:
            visible_ids : (N_vis,)  indici patch visibili (sorted)
            mask_ids    : (N_mask,) indici patch mascherate (sorted)
        """
        # Unione di tutti i blocchi campionati
        combined_mask = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)

        # Campiona i blocchi e accumula l'unione
        blocks = [self._sample_block() for _ in range(self.num_blocks)]
        for block in blocks:
            combined_mask |= block

        # Fallback: se troppo poche patch visibili, rimuovi blocchi uno alla volta
        # finché non si rispetta min_visible
        while combined_mask.sum() > (self.num_patches - self.min_visible) and len(blocks) > 1:
            blocks.pop()
            combined_mask = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
            for block in blocks:
                combined_mask |= block

        # Flatten e split in visible / masked
        flat_mask = combined_mask.flatten()            # (N,) bool
        all_ids   = torch.arange(self.num_patches)

        mask_ids    = all_ids[flat_mask]               # patch mascherate
        visible_ids = all_ids[~flat_mask]              # patch visibili

        # Shuffle per evitare bias posizionale nel training
        visible_ids = visible_ids[torch.randperm(len(visible_ids))]
        mask_ids    = mask_ids[torch.randperm(len(mask_ids))]

        return visible_ids, mask_ids