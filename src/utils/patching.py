import torch

def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Converte un batch di immagini in una sequenza di patch appiattite.
    Utile per generare i target per la loss MSE nel MAE/VAE.
    
    Input:  (B, C, H, W)
    Output: (B, N, patch_size**2 * C) dove N = (H/patch_size) * (W/patch_size)
    """
    B, C, H, W = imgs.shape
    assert H == W and H % patch_size == 0, "L'immagine deve essere quadrata e divisibile per patch_size"
    
    grid_size = H // patch_size
    
    # Reshape: (B, C, H_grid, patch_size, W_grid, patch_size)
    x = imgs.reshape(B, C, grid_size, patch_size, grid_size, patch_size)
    
    # Trasposizione: (B, H_grid, W_grid, patch_size, patch_size, C)
    x = torch.einsum('nchpwq->nhwpqc', x)
    
    # Flatten: (B, H_grid * W_grid, patch_size**2 * C)
    x = x.reshape(B, grid_size * grid_size, patch_size**2 * C)
    
    return x

def unpatchify(x: torch.Tensor, patch_size: int = 16, in_chans: int = 3) -> torch.Tensor:
    """
    Converte una sequenza di patch (output del decoder) di nuovo in un'immagine.
    Utile per la visualizzazione e il salvataggio dei risultati.
    
    Input:  (B, N, patch_size**2 * in_chans)
    Output: (B, in_chans, H, W)
    """
    B = x.shape[0]
    grid_size = int(x.shape[1] ** 0.5)
    assert grid_size * grid_size == x.shape[1], "Il numero di patch non forma un quadrato perfetto."

    # Reshape: (B, H_grid, W_grid, patch_size, patch_size, C)
    x = x.reshape(B, grid_size, grid_size, patch_size, patch_size, in_chans)
    
    # Trasposizione: (B, C, H_grid, patch_size, W_grid, patch_size)
    x = torch.einsum('nhwpqc->nchpwq', x)
    
    # Merge spaziale: (B, C, H_grid * patch_size, W_grid * patch_size)
    imgs = x.reshape(B, in_chans, grid_size * patch_size, grid_size * patch_size)
    
    return imgs