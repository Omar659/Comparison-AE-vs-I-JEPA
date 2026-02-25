import os
import argparse
import torch

from src.constants import (
    DEVICE, DATA_DIR, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES,
    LINEAR_PROBE_EPOCHS, LINEAR_PROBE_LR, VAE_LATENT_DIM
)
from src.dataset import get_dataloaders
from src.models.vit import VisionTransformerEncoder
from src.models.decoder import VisionTransformerDecoder
from src.models.vae import VariationalMaskedAE
from src.utils.linear_probe import linear_probe

def parse_args():
    parser = argparse.ArgumentParser(description="Confronto Linear Probe I-JEPA vs VAE")
    parser.add_argument("--ijepa_ckpt", type=str, required=True,
                        help="Path al checkpoint del miglior modello I-JEPA (es. checkpoints/ijepa/0/best_model.pt)")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path al checkpoint del miglior modello VAE (es. checkpoints/vae/0/best_model.pt)")
    parser.add_argument("--epochs", type=int, default=LINEAR_PROBE_EPOCHS,
                        help="Epoche di training per il classificatore lineare")
    return parser.parse_args()

def load_ijepa_encoder(ckpt_path: str) -> VisionTransformerEncoder:
    """Carica il Target Encoder di I-JEPA dal checkpoint."""
    print(f"Caricamento I-JEPA Target Encoder da: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint I-JEPA non trovato: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    encoder = VisionTransformerEncoder().to(DEVICE)
    
    # I-JEPA salva 'encoder', 'target_encoder' e 'predictor'.
    # Usiamo il 'target_encoder' che tipicamente produce le feature semantiche migliori.
    encoder.load_state_dict(ckpt['states']['target_encoder'])
    encoder.eval()
    
    # Congela i pesi per sicurezza
    for p in encoder.parameters():
        p.requires_grad = False
        
    return encoder

def load_vae_encoder(ckpt_path: str) -> VisionTransformerEncoder:
    """Carica l'Encoder del VAE dal checkpoint."""
    print(f"Caricamento VAE Encoder da: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint VAE non trovato: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Per caricare i pesi dobbiamo istanziare l'intero VAE e poi estrarre l'encoder
    enc = VisionTransformerEncoder()
    dec = VisionTransformerDecoder(embed_dim=VAE_LATENT_DIM)
    vae = VariationalMaskedAE(enc, dec)
    
    vae.load_state_dict(ckpt['states']['model'])
    
    encoder = vae.encoder.to(DEVICE)
    encoder.eval()
    
    # Congela i pesi
    for p in encoder.parameters():
        p.requires_grad = False
        
    return encoder

def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  LINEAR PROBE COMPARISON: I-JEPA vs VAE")
    print("=" * 60)
    
    # 1. Caricamento Dataset
    print("\n[1/4] Caricamento Dataset (ImageNet-100)...")
    train_loader, _, test_loader = get_dataloaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    print(f"      Train set: {len(train_loader.dataset)} immagini")
    print(f"      Test set:  {len(test_loader.dataset)} immagini")

    # 2. Valutazione I-JEPA
    print(f"\n[2/4] Avvio Linear Probe per I-JEPA ({args.epochs} epoche)...")
    ijepa_encoder = load_ijepa_encoder(args.ijepa_ckpt)
    ijepa_results = linear_probe(
        encoder=ijepa_encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        epochs=args.epochs,
        lr=LINEAR_PROBE_LR
    )
    
    # Libera memoria
    del ijepa_encoder
    torch.cuda.empty_cache()

    # 3. Valutazione VAE
    print(f"\n[3/4] Avvio Linear Probe per VAE ({args.epochs} epoche)...")
    vae_encoder = load_vae_encoder(args.vae_ckpt)
    vae_results = linear_probe(
        encoder=vae_encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        epochs=args.epochs,
        lr=LINEAR_PROBE_LR
    )
    
    del vae_encoder
    torch.cuda.empty_cache()

    # 4. Report Finale
    print("\n[4/4] RISULTATI FINALI")
    print("=" * 60)
    
    ijepa_top1 = ijepa_results['top1'] * 100
    ijepa_top5 = ijepa_results['top5'] * 100
    vae_top1   = vae_results['top1'] * 100
    vae_top5   = vae_results['top5'] * 100
    
    print(f"  [I-JEPA] Top-1 Acc: {ijepa_top1:.2f}%  |  Top-5 Acc: {ijepa_top5:.2f}%")
    print(f"  [VAE]    Top-1 Acc: {vae_top1:.2f}%  |  Top-5 Acc: {vae_top5:.2f}%")
    print("-" * 60)
    
    if ijepa_top1 > vae_top1:
        diff = ijepa_top1 - vae_top1
        print(f"  VINCITORE SULLA SEMANTICA: I-JEPA (+{diff:.2f}%)")
    elif vae_top1 > ijepa_top1:
        diff = vae_top1 - ijepa_top1
        print(f"  VINCITORE SULLA SEMANTICA: VAE (+{diff:.2f}%)")
    else:
        print("  PAREGGIO!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()