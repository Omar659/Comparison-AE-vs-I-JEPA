"""
Training loop per Variational Masked Autoencoder (VAE).

Flusso per ogni batch:
  1. RandomMaskGenerator genera visible_ids e mask_ids (75% mascherato, stile MAE)
  2. Forward VAE(imgs, visible_ids) -> pred_patches, mu, logvar
  3. Loss: MSE solo sulle patch mascherate + get_kl_weight(epoch) * KL
  4. Backward + optimizer step
  5. LR schedule + weight decay schedule

Valutazione:
  - Validation Loss (MSE + KL) ogni epoca
  - Test: MSE e PSNR sulle sole patch mascherate (le visibili sono ground truth)

Uso:
    python train_vae.py
    python train_vae.py --resume checkpoints/vae/0/checkpoint_epoch_010.pt
    python train_vae.py --no-wandb
"""

import os
import json
import argparse
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.constants import (
    SEED, DEVICE, USE_BFLOAT16,
    DATA_DIR, BATCH_SIZE, NUM_WORKERS,
    EMBED_DIM, DECODER_EMBED_DIM,
    VAE_LATENT_DIM, KLD_WEIGHT,
    EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR,
    WEIGHT_DECAY_START, WEIGHT_DECAY_END, GRAD_CLIP,
    CHECKPOINT_DIR_VAE, CHECKPOINT_EVERY, RESULTS_DIR_VAE,
    WANDB_PROJECT, WANDB_RUN_NAME, PATCH_SIZE,
    NUM_MASK_BLOCKS, MASK_BLOCK_SCALE,
)
from src.dataset import get_dataloaders
from src.models.vit import VisionTransformerEncoder
from src.models.decoder import VisionTransformerDecoder
from src.models.vae import VariationalMaskedAE
from src.masks.random_mask import RandomMaskGenerator
from src.utils.patching import patchify
from src.utils.training import (
    get_lr, get_weight_decay,
    save_checkpoint, load_checkpoint, save_results, get_kl_weight,
)


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Variational Masked Autoencoder")
    parser.add_argument("--resume",   type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


# =============================================================================
# TRAIN
# =============================================================================

def train(
    model:             VariationalMaskedAE,
    loader:            torch.utils.data.DataLoader,
    optimizer:         torch.optim.Optimizer,
    scaler:            torch.amp.GradScaler,
    mask_gen:          RandomMaskGenerator,
    device:            torch.device,
    epoch:             int,
    global_step:       int,
    use_wandb:         bool,
    current_kl_weight: float,
) -> tuple[float, float, float, int]:

    model.train()
    total_loss = total_recon = total_kl = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", ncols=110, leave=False)

    for batch_idx, (imgs, _) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        B    = imgs.shape[0]

        visible_ids, mask_ids = mask_gen(batch_size=B)
        visible_ids = visible_ids.to(device)
        mask_ids    = mask_ids.to(device)

        target_patches = patchify(imgs, patch_size=PATCH_SIZE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
            pred_patches, mu, logvar = model(imgs, visible_ids)

            # Loss MSE solo sulle patch mascherate
            D = pred_patches.shape[-1]
            idx = mask_ids.unsqueeze(-1).expand(-1, -1, D)
            recon_loss = F.mse_loss(
                torch.gather(pred_patches, 1, idx),
                torch.gather(target_patches, 1, idx),
            )

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            loss    = recon_loss + current_kl_weight * kl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()
        num_batches += 1
        global_step += 1

        pbar.set_postfix({
            'loss':  f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl':    f'{kl_loss.item():.4f}',
            'lr':    f'{get_lr(epoch, WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR):.1e}',
        })

        if use_wandb:
            import wandb
            wandb.log({
                'train/loss_step':  loss.item(),
                'train/recon_step': recon_loss.item(),
                'train/kl_step':    kl_loss.item(),
                'step/global':      global_step,
                'step/epoch':       batch_idx + 1,
                'epoch':            epoch,
            })

    n = max(1, num_batches)
    return total_loss / n, total_recon / n, total_kl / n, global_step


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validation(
    model:             VariationalMaskedAE,
    loader:            torch.utils.data.DataLoader,
    mask_gen:          RandomMaskGenerator,
    device:            torch.device,
    current_kl_weight: float,
) -> tuple[float, float, float]:

    model.eval()
    total_loss = total_recon = total_kl = 0.0
    num_batches = 0

    for imgs, _ in tqdm(loader, desc='Validation', ncols=100, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        B    = imgs.shape[0]

        visible_ids, mask_ids = mask_gen(batch_size=B)
        visible_ids = visible_ids.to(device)
        mask_ids    = mask_ids.to(device)
        target_patches = patchify(imgs, patch_size=PATCH_SIZE)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
            pred_patches, mu, logvar = model(imgs, visible_ids)

            D   = pred_patches.shape[-1]
            idx = mask_ids.unsqueeze(-1).expand(-1, -1, D)
            recon_loss = F.mse_loss(
                torch.gather(pred_patches, 1, idx),
                torch.gather(target_patches, 1, idx),
            )
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            loss    = recon_loss + current_kl_weight * kl_loss

        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()
        num_batches += 1

    n = max(1, num_batches)
    return total_loss / n, total_recon / n, total_kl / n


# =============================================================================
# TEST
# =============================================================================

@torch.no_grad()
def test(
    model:    VariationalMaskedAE,
    loader:   torch.utils.data.DataLoader,
    mask_gen: RandomMaskGenerator,
    device:   torch.device,
) -> tuple[float, float]:
    """
    MSE e PSNR sul test set.
    Le patch visibili vengono sostituite con il ground truth prima del MSE,
    così la metrica misura solo la qualità di ricostruzione delle patch mascherate.
    """
    model.eval()
    total_mse   = 0.0
    num_batches = 0

    for imgs, _ in tqdm(loader, desc='Test', ncols=100, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        B    = imgs.shape[0]

        visible_ids, mask_ids = mask_gen(batch_size=B)
        visible_ids = visible_ids.to(device)
        mask_ids    = mask_ids.to(device)
        target_patches = patchify(imgs, patch_size=PATCH_SIZE)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_BFLOAT16):
            pred_patches, _, _ = model(imgs, visible_ids)

        # Sostituisce le patch visibili con il ground truth (valuta solo quelle mascherate)
        D   = pred_patches.shape[-1]
        idx = visible_ids.unsqueeze(-1).expand(-1, -1, D)
        pred_patches = pred_patches.clone()
        pred_patches.scatter_(1, idx, torch.gather(target_patches, 1, idx).to(pred_patches.dtype))

        mse = F.mse_loss(pred_patches, target_patches).item()
        total_mse   += mse
        num_batches += 1

    avg_mse = total_mse / max(1, num_batches)
    psnr    = 10 * math.log10(25.0 / avg_mse) if avg_mse > 0 else 100.0  # MAX^2 = 5^2 = 25
    return avg_mse, psnr


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR_VAE(), exist_ok=True)
    os.makedirs(RESULTS_DIR_VAE(),    exist_ok=True)

    listdir = [int(x) for x in os.listdir(RESULTS_DIR_VAE()) if x.isdigit()]
    run_path_name = str(max(listdir)) if listdir else "-1"

    start_epoch    = 0
    global_step    = 0
    all_results    = []
    resumed_run_id = None
    ckpt           = None

    if args.resume:
        ckpt           = load_checkpoint(args.resume, DEVICE)
        start_epoch    = ckpt['epoch'] + 1
        resumed_run_id = ckpt.get('wandb_run_id', None)
        results_path   = os.path.join(RESULTS_DIR_VAE(run_path_name), "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results = json.load(f)
    else:
        run_path_name = str(int(run_path_name) + 1)
        os.makedirs(CHECKPOINT_DIR_VAE(run_path_name), exist_ok=True)
        os.makedirs(RESULTS_DIR_VAE(run_path_name),    exist_ok=True)

    use_wandb = not args.no_wandb
    run_name  = WANDB_RUN_NAME or f"vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if use_wandb:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name if not resumed_run_id else None,
            id=resumed_run_id,
            resume="must" if resumed_run_id else None,
            config={
                'model_type':       'VAE_Masked',
                'embed_dim':        EMBED_DIM,
                'decoder_dim':      DECODER_EMBED_DIM,
                'latent_dim':       VAE_LATENT_DIM,
                'kld_weight':       KLD_WEIGHT,
                'num_mask_blocks':  NUM_MASK_BLOCKS,
                'mask_block_scale': MASK_BLOCK_SCALE,
                'epochs':           EPOCHS,
                'batch_size':       BATCH_SIZE,
                'base_lr':          BASE_LR,
            },
        )

    print("Caricamento dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    steps_per_epoch = len(train_loader)

    print("Inizializzazione modello VAE...")
    encoder = VisionTransformerEncoder()
    decoder = VisionTransformerDecoder(embed_dim=VAE_LATENT_DIM)
    model   = VariationalMaskedAE(encoder, decoder).to(DEVICE)
    print(f"  Parametri trainabili: {sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=BASE_LR, weight_decay=WEIGHT_DECAY_START, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_BFLOAT16)

    if ckpt is not None:
        model.load_state_dict(ckpt['states']['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        global_step = start_epoch * steps_per_epoch
        print(f"  Ripreso dall'epoca {start_epoch}")

    mask_gen   = RandomMaskGenerator()
    best_psnr  = 0.0
    best_epoch = 0

    print(f"\nInizio training VAE - {EPOCHS} epoche su {DEVICE}")
    print("=" * 60)

    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc='Training VAE', ncols=100)

    for epoch in epoch_pbar:
        epoch_start = time.time()

        lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)
        wd = get_weight_decay(epoch, EPOCHS, WEIGHT_DECAY_START, WEIGHT_DECAY_END)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            pg['weight_decay'] = wd

        current_kl_weight = get_kl_weight(epoch, WARMUP_EPOCHS, KLD_WEIGHT)

        train_loss, train_recon, train_kl, global_step = train(
            model, train_loader, optimizer, scaler, mask_gen,
            DEVICE, epoch, global_step, use_wandb, current_kl_weight,
        )
        val_loss, val_recon, val_kl = validation(
            model, val_loader, mask_gen, DEVICE, current_kl_weight,
        )
        test_mse, test_psnr = test(model, test_loader, mask_gen, DEVICE)

        epoch_time = time.time() - epoch_start

        epoch_metrics = {
            'epoch':       epoch,
            'train_loss':  round(train_loss,  6),
            'train_recon': round(train_recon, 6),
            'train_kl':    round(train_kl,    6),
            'val_loss':    round(val_loss,    6),
            'val_recon':   round(val_recon,   6),
            'val_kl':      round(val_kl,      6),
            'test_mse':    round(test_mse,    6),
            'test_psnr':   round(test_psnr,   2),
            'kl_weight':   round(current_kl_weight, 6),
            'lr':          round(lr,          8),
            'epoch_time':  round(epoch_time,  2),
        }

        epoch_pbar.set_postfix({
            'T_Loss':   f'{train_loss:.4f}',
            'V_Loss':   f'{val_loss:.4f}',
            'TestPSNR': f'{test_psnr:.1f}dB',
        })
        tqdm.write(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"T_Loss: {train_loss:.4f} (Rec:{train_recon:.4f} KL:{train_kl:.2f}) | "
            f"V_Loss: {val_loss:.4f} | "
            f"PSNR: {test_psnr:.2f}dB | "
            f"kl_w: {current_kl_weight:.5f} | lr: {lr:.2e} | {epoch_time:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                'train/loss_epoch':  train_loss,
                'train/recon_loss':  train_recon,
                'train/kl_loss':     train_kl,
                'val/loss':          val_loss,
                'val/recon_loss':    val_recon,
                'val/kl_loss':       val_kl,
                'eval/test_mse':     test_mse,
                'eval/test_psnr':    test_psnr,
                'optim/lr':          lr,
                'optim/kl_weight':   current_kl_weight,
                'epoch':             epoch,
            })

        all_results.append(epoch_metrics)
        save_results(all_results, os.path.join(RESULTS_DIR_VAE(run_path_name), "results.json"))

        if test_psnr > best_psnr:
            best_psnr  = test_psnr
            best_epoch = epoch
            save_checkpoint(
                path=os.path.join(CHECKPOINT_DIR_VAE(run_path_name), "best_model.pt"),
                epoch=epoch, states={'model': model.state_dict()},
                optimizer=optimizer, scaler=scaler, metrics=epoch_metrics,
            )
            tqdm.write(f"  [best] Nuovo best PSNR: {test_psnr:.2f}dB (epoca {epoch})")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                path=os.path.join(CHECKPOINT_DIR_VAE(run_path_name), f"checkpoint_epoch_{epoch:03d}.pt"),
                epoch=epoch, states={'model': model.state_dict()},
                optimizer=optimizer, scaler=scaler, metrics=epoch_metrics,
            )

    save_checkpoint(
        path=os.path.join(CHECKPOINT_DIR_VAE(run_path_name), "final_model.pt"),
        epoch=EPOCHS - 1, states={'model': model.state_dict()},
        optimizer=optimizer, scaler=scaler, metrics=all_results[-1],
    )
    save_results({
        'run_name':        run_name,
        'total_epochs':    EPOCHS,
        'best_psnr':       best_psnr,
        'best_epoch':      best_epoch,
        'final_test_mse':  all_results[-1]['test_mse'],
        'final_test_psnr': all_results[-1]['test_psnr'],
    }, os.path.join(RESULTS_DIR_VAE(run_path_name), "summary.json"))

    print("\n" + "=" * 60)
    print(f"Training VAE completato!")
    print(f"  Best PSNR:  {best_psnr:.2f}dB (epoca {best_epoch})")
    print(f"  Checkpoint: {CHECKPOINT_DIR_VAE(run_path_name)}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()