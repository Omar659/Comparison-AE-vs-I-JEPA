"""
generate_vae.py

Visualizzazione dei risultati del Variational Masked Autoencoder (VAE).
Mostra:
 1. Immagine Originale
 2. Immagine Mascherata (Input al modello, 75% nascosto)
 3. Ricostruzione (Inpainting) con il fix per i pixel visibili
 4. Generazione da zero (Puro Rumore Latente)

Uso:
    python generate_vae.py --ckpt checkpoints/vae/0/final_model.pt
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.constants import (
    DEVICE, DATA_DIR, PATCH_SIZE, VAE_LATENT_DIM
)
from src.dataset import get_dataloaders
from src.models.vit import VisionTransformerEncoder
from src.models.decoder import VisionTransformerDecoder
from src.models.vae import VariationalMaskedAE
from src.utils.patching import patchify, unpatchify
from src.masks.random_mask import RandomMaskGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path al checkpoint del VAE (es. results/vae/0/final_model.pt)")
    parser.add_argument("--save_path", type=str, default=".", help="Numero run dove salvare il risultato es. \"1\"")
    parser.add_argument("--n_samples", type=int, default=10, help="Numero di immagini da generare")
    return parser.parse_args()

def denormalize(tensor):
    """Annulla la normalizzazione di ImageNet per la visualizzazione."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def load_model(ckpt_path):
    print(f"Caricamento VAE da: {ckpt_path}")
    enc = VisionTransformerEncoder()
    dec = VisionTransformerDecoder(embed_dim=VAE_LATENT_DIM)
    vae = VariationalMaskedAE(enc, dec).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if 'states' in ckpt and 'model' in ckpt['states']:
        vae.load_state_dict(ckpt['states']['model'])
    elif 'model_state_dict' in ckpt:
        vae.load_state_dict(ckpt['model_state_dict'])
    else:
        vae.load_state_dict(ckpt)
        
    vae.eval()
    return vae

def main():
    args = parse_args()
    vae = load_model(args.ckpt)
    
    # 1. Caricamento Dati
    _, _, test_loader = get_dataloaders(DATA_DIR, batch_size=args.n_samples, num_workers=0)
    imgs, _ = next(iter(test_loader))
    imgs = imgs.to(DEVICE)
    B = imgs.shape[0]
    
    # Inizializziamo il generatore di maschere (usa le costanti, es. ratio=0.75)
    mask_gen = RandomMaskGenerator(mask_ratio=0.50)
    visible_ids, mask_ids = mask_gen(batch_size=B)
    visible_ids = visible_ids.to(DEVICE)
    mask_ids = mask_ids.to(DEVICE)
    
    N_total = mask_gen.num_patches
    N_vis = mask_gen.num_vis

    print("Generazione in corso...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            
            # ==========================================================
            # TASK A: INPAINTING (Ricostruzione dei buchi)
            # ==========================================================
            pred_patches, _, _ = vae(imgs, visible_ids, mask_ids)
            
            target_patches = patchify(imgs, patch_size=PATCH_SIZE)
            D = pred_patches.shape[-1]
            vis_idx_expanded = visible_ids.unsqueeze(-1).expand(-1, -1, D)
            
            # Sovrascriviamo le predizioni con i target nelle posizioni visibili
            pred_patches.scatter_(
                dim=1, 
                index=vis_idx_expanded, 
                src=torch.gather(target_patches, dim=1, index=vis_idx_expanded)
            )
            
            imgs_inpainting = unpatchify(pred_patches)

            # ==========================================================
            # TASK B: GENERAZIONE DA PURO RUMORE
            # ==========================================================
            # Campioniamo rumore SOLO per il numero di token visibili (es. 49)
            z_random = torch.randn(B, N_vis, VAE_LATENT_DIM).to(DEVICE)
            
            # Creiamo posizioni casuali per il Decoder
            noise = torch.rand(B, N_total).to(DEVICE)
            ids_restore_dummy = torch.argsort(noise, dim=1)
            
            pred_patches_gen = vae.decoder(z_random, ids_restore=ids_restore_dummy)
            imgs_gen = unpatchify(pred_patches_gen)

    # ==========================================================
    # PREPARAZIONE PLOT
    # ==========================================================
    # Creiamo l'immagine con i "buchi neri" per mostrare cosa vede il modello
    imgs_masked_plot = imgs.clone()
    H = W = int(N_total ** 0.5)
    for b in range(B):
        # Mettiamo a 0 (nero) le patch mascherate
        for idx in mask_ids[b].tolist():
            r, c = idx // W, idx % W
            pr, pc = r * PATCH_SIZE, c * PATCH_SIZE
            imgs_masked_plot[b, :, pr:pr+PATCH_SIZE, pc:pc+PATCH_SIZE] = 0.0

    # Denormalizziamo e prepariamo per matplotlib
    imgs             = denormalize(imgs).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    imgs_masked_plot = denormalize(imgs_masked_plot).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    imgs_inpainting  = denormalize(imgs_inpainting).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    imgs_gen         = denormalize(imgs_gen).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    # --- Plot ---
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    cols = ["Originale", "Input (75% Nascosto)", "Inpainting (Ricostruzione)", "Generazione Random"]
    
    if B == 1:
        axes = axes[None, :]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')
        
    for i in range(B):
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(imgs_masked_plot[i])
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(imgs_inpainting[i])
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(imgs_gen[i])
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig("./results/vae/" + args.save_path + "/vae_results.png", dpi=150, bbox_inches='tight')
    print(f"\nRisultati salvati con successo in: {args.save_path}")

if __name__ == "__main__":
    main()