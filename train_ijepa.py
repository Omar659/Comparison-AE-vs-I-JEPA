"""
Training loop per Image-JEPA stile V-JEPA 2.

Flusso per ogni batch:
  1. MaskGenerator genera visible_ids e mask_ids
  2. Forward IJEPA -> loss L1
  3. Backward + optimizer step
  4. EMA update del target encoder
  5. LR schedule + weight decay schedule

Valutazione:
  - k-NN accuracy ogni epoca (sul test set)
  - Linear probe ogni LINEAR_PROBE_EVERY epoche
  - Loss di validation ogni epoca

Logging:
  - W&B: tutte le metriche in tempo reale
  - JSON: risultati per epoca in results/ijepa/results.json
  - Checkpoint: ogni CHECKPOINT_EVERY epoche + best model + finale

Uso:
    python train_ijepa.py
    python train_ijepa.py --resume checkpoints/ijepa/checkpoint_epoch_010.pt
    python train_ijepa.py --no-wandb
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.constants import (
    SEED, DEVICE, USE_BFLOAT16,
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES,
    EMBED_DIM, ENCODER_DEPTH, NUM_HEADS, MLP_RATIO, DROP, ATTN_DROP,
    PRED_EMBED_DIM, PRED_DEPTH, PRED_NUM_HEADS,
    NUM_MASK_BLOCKS, MASK_BLOCK_SCALE, MASK_ASPECT_RATIO, MIN_VISIBLE_PATCHES,
    EMA_MOMENTUM_START, EMA_MOMENTUM_END,
    EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR,
    WEIGHT_DECAY_START, WEIGHT_DECAY_END, GRAD_CLIP,
    KNN_K, KNN_TEMPERATURE, LINEAR_PROBE_EPOCHS, LINEAR_PROBE_LR, LINEAR_PROBE_EVERY,
    CHECKPOINT_DIR_IJEPA, CHECKPOINT_EVERY, RESULTS_DIR_IJEPA, WANDB_PROJECT, WANDB_RUN_NAME,
)
from src.dataset import get_dataloaders
from src.models.vit import VisionTransformerEncoder
from src.models.predictor import Predictor
from src.models.ijepa import IJEPA
from src.masks.mask_generator import MaskGenerator
from src.utils.ema import update_ema, get_momentum
from src.utils.knn import knn_accuracy
from src.utils.linear_probe import linear_probe
from src.utils.training import (
    get_lr, get_weight_decay,
    save_checkpoint, load_checkpoint,
    save_results,
)


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Image-JEPA")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path checkpoint da cui riprendere il training")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disabilita W&B (utile per debug locale)")
    return parser.parse_args()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    model:       IJEPA,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    scaler:      torch.amp.GradScaler,
    mask_gen:    MaskGenerator,
    device:      torch.device,
    epoch:       int,
    total_steps: int,
    global_step: int,
    use_wandb:   bool,
) -> tuple[float, int]:
    """
    Esegue un'epoca di training.

    Returns:
        avg_loss    : loss media sull'epoca
        global_step : step globale aggiornato
    """
    model.encoder.train()
    model.predictor.train()
    model.target_encoder.eval()

    total_loss      = 0.0
    num_batches     = 0
    steps_per_epoch = len(loader)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [train]",
        ncols=100,
        leave=False,
    )

    for batch_idx, (imgs, _) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        B    = imgs.shape[0]

        # Genera maschere
        visible_ids, mask_ids = mask_gen(batch_size=B)
        visible_ids = visible_ids.to(device)
        mask_ids    = mask_ids.to(device)

        # Forward + Loss
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                 enabled=USE_BFLOAT16):
            loss, _ = model(imgs, visible_ids, mask_ids)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # EMA update
        momentum = get_momentum(global_step, total_steps,
                                EMA_MOMENTUM_START, EMA_MOMENTUM_END)
        update_ema(model.encoder, model.target_encoder, momentum)

        total_loss  += loss.item()
        num_batches += 1
        global_step += 1

        # Aggiorna barra con metriche correnti
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg':  f'{total_loss/num_batches:.4f}',
            'lr':   f'{get_lr(epoch, WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR):.1e}',
            'mom':  f'{momentum:.4f}',
        })

        if use_wandb:
            import wandb
            wandb.log({
                'train/loss_step':   loss.item(),
                'train/momentum':    momentum,
                'step/global':       global_step,
                'step/epoch':        batch_idx + 1,
                'step/epoch_total':  steps_per_epoch,
                'step/progress_pct': (batch_idx + 1) / steps_per_epoch * 100,
                'epoch':             epoch,
            })

    return total_loss / max(1, num_batches), global_step


@torch.no_grad()
def validation(
    model:    IJEPA,
    loader:   torch.utils.data.DataLoader,
    mask_gen: MaskGenerator,
    device:   torch.device,
) -> float:
    """Calcola la loss di validation."""
    model.encoder.eval()
    model.predictor.eval()

    total_loss  = 0.0
    num_batches = 0

    for imgs, _ in tqdm(loader, desc='Validation', ncols=100, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        B    = imgs.shape[0]

        visible_ids, mask_ids = mask_gen(batch_size=B)
        visible_ids = visible_ids.to(device)
        mask_ids    = mask_ids.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16,
                                 enabled=USE_BFLOAT16):
            loss, _ = model(imgs, visible_ids, mask_ids)

        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)

def test(
    model:        IJEPA,
    train_loader: torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
    epoch:        int,
    device:   torch.device,
) -> tuple[float, Optional[float]]:
    """Calcola due metriche con il testset, knn accuracy e il linear probe con top1 e top5, 
    questo solo ogni LINEAR_PROBE_EVERY epoch"""
    
    # k-NN accuracy
    knn_acc = knn_accuracy(
        model.target_encoder,
        train_loader, test_loader,
        DEVICE, k=KNN_K, temperature=KNN_TEMPERATURE,
    )
    
    lp = None    
    # Linear Probe ogni LINEAR_PROBE_EVERY epoche
    if (epoch + 1) % LINEAR_PROBE_EVERY == 0 or epoch == EPOCHS - 1:
        lp = linear_probe(
            model.target_encoder,
            train_loader, test_loader,
            DEVICE, NUM_CLASSES,
            epochs=LINEAR_PROBE_EPOCHS,
            lr=LINEAR_PROBE_LR,
        )
    return knn_acc, lp

# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR_IJEPA(), exist_ok=True)
    os.makedirs(RESULTS_DIR_IJEPA(),    exist_ok=True)
    
    listdir = os.listdir(RESULTS_DIR_IJEPA())
    run_path_name = "-1"
    if listdir != []:
        listdir = [int(x) for x in listdir]
        listdir.sort()
        run_path_name = str(listdir[-1])

    # --- Resume: carica checkpoint prima di inizializzare W&B ---
    start_epoch    = 0
    global_step    = 0
    all_results    = []
    resumed_run_id = None    
    ckpt           = None

    if args.resume:
        ckpt           = load_checkpoint(args.resume, DEVICE)
        start_epoch    = ckpt['epoch'] + 1
        resumed_run_id = ckpt.get('wandb_run_id', None)
        results_path   = os.path.join(RESULTS_DIR_IJEPA(run_path_name), "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results = json.load(f)
    else:
        run_path_name = str(int(run_path_name) + 1)
        os.makedirs(CHECKPOINT_DIR_IJEPA(run_path_name), exist_ok=True)
        os.makedirs(RESULTS_DIR_IJEPA(run_path_name),    exist_ok=True)
    
    # --- W&B ---
    use_wandb = not args.no_wandb
    run_name  = WANDB_RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_wandb:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name if not resumed_run_id else None,
            id=resumed_run_id,
            resume="must" if resumed_run_id else None,
            config={
                'embed_dim':          EMBED_DIM,
                'encoder_depth':      ENCODER_DEPTH,
                'num_heads':          NUM_HEADS,
                'pred_embed_dim':     PRED_EMBED_DIM,
                'pred_depth':         PRED_DEPTH,
                'num_mask_blocks':    NUM_MASK_BLOCKS,
                'mask_block_scale':   MASK_BLOCK_SCALE,
                'mask_aspect_ratio':  MASK_ASPECT_RATIO,
                'epochs':             EPOCHS,
                'warmup_epochs':      WARMUP_EPOCHS,
                'base_lr':            BASE_LR,
                'min_lr':             MIN_LR,
                'weight_decay_start': WEIGHT_DECAY_START,
                'weight_decay_end':   WEIGHT_DECAY_END,
                'batch_size':         BATCH_SIZE,
                'ema_start':          EMA_MOMENTUM_START,
                'ema_end':            EMA_MOMENTUM_END,
                'grad_clip':          GRAD_CLIP,
            },
        )

    # --- Dataset ---
    print("Caricamento dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # --- Modello ---
    print("Inizializzazione modello...")
    encoder   = VisionTransformerEncoder().to(DEVICE)
    predictor = Predictor().to(DEVICE)
    model     = IJEPA(encoder, predictor).to(DEVICE)
    print(f"  Parametri trainabili: "
          f"{sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M")

    # --- Optimizer & Scaler ---
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY_START,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_BFLOAT16)

    # --- Carica pesi se resume ---
    if ckpt is not None:
        model.encoder.load_state_dict(ckpt['states']['encoder'])
        model.target_encoder.load_state_dict(ckpt['states']['target_encoder'])
        model.predictor.load_state_dict(ckpt['states']['predictor'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        global_step = start_epoch * steps_per_epoch
        print(f"  Ripreso dall'epoca {start_epoch}")

    # --- Mask Generator ---
    mask_gen = MaskGenerator()

    # --- Tracking best model ---
    best_knn_acc = 0.0
    best_epoch   = 0

    # ==========================================================================
    # LOOP DI TRAINING
    # ==========================================================================
    print(f"\nInizio training - {EPOCHS} epoche su {DEVICE}")
    print("=" * 60)

    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc='Training', ncols=100)

    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Schedule LR e Weight Decay
        lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)
        wd = get_weight_decay(epoch, EPOCHS, WEIGHT_DECAY_START, WEIGHT_DECAY_END)
        for pg in optimizer.param_groups:
            pg['lr']           = lr
            pg['weight_decay'] = wd

        # Training
        train_loss, global_step = train(
            model, train_loader, optimizer, scaler, mask_gen,
            DEVICE, epoch, total_steps, global_step, use_wandb,
        )

        # Validation
        val_loss = validation(model, val_loader, mask_gen, DEVICE)
                
        # Test
        knn_acc, lp = test(model, train_loader, test_loader, epoch, DEVICE)
        
        epoch_time = time.time() - epoch_start

        # Metriche epoca
        epoch_metrics = {
            'epoch':        epoch,
            'train_loss':   round(train_loss, 6),
            'val_loss':     round(val_loss,   6),
            'knn_acc':      round(knn_acc,    6),
            'lr':           round(lr,         8),
            'weight_decay': round(wd,         6),
            'ema_momentum': round(get_momentum(
                global_step, total_steps, EMA_MOMENTUM_START, EMA_MOMENTUM_END), 6),
            'epoch_time_s': round(epoch_time, 2),
        }
        
        if lp is not None:
            epoch_metrics['linear_probe_top1'] = round(lp['top1'], 6)
            epoch_metrics['linear_probe_top5'] = round(lp['top5'], 6)

        # Aggiorna barra epoche
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val':   f'{val_loss:.4f}',
            'knn':   f'{knn_acc*100:.1f}%',
            'lr':    f'{lr:.1e}',
        })

        # Stampa riepilogo epoca
        tqdm.write(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"knn: {knn_acc*100:.2f}% | "
            f"lr: {lr:.2e} | {epoch_time:.1f}s"
        )

        # W&B logging epoca
        if use_wandb:
            import wandb
            wandb.log({
                'train/loss_epoch': train_loss,
                'val/loss':         val_loss,
                'eval/knn_acc':     knn_acc,
                'optim/lr':         lr,
                'optim/wd':         wd,
                'epoch':            epoch,
            })
            
            if lp is not None:
                wandb.log({
                    'eval/linear_probe_top1': lp['top1'],
                    'eval/linear_probe_top5': lp['top5'],
                    'epoch': epoch,
                })

        # Salva risultati JSON
        all_results.append(epoch_metrics)
        save_results(all_results, os.path.join(RESULTS_DIR_IJEPA(run_path_name), "results.json"))

        # Checkpoint: best model
        if knn_acc > best_knn_acc:
            best_knn_acc = knn_acc
            best_epoch   = epoch
            save_checkpoint(
                path=os.path.join(CHECKPOINT_DIR_IJEPA(run_path_name), "best_model.pt"),
                epoch=epoch,
                states={
                    'encoder':        model.encoder.state_dict(),
                    'target_encoder': model.target_encoder.state_dict(),
                    'predictor':      model.predictor.state_dict(),
                },
                optimizer=optimizer,
                scaler=scaler,
                metrics=epoch_metrics,
            )
            print(f"  [best] Nuovo best k-NN: {knn_acc*100:.2f}% (epoca {epoch})")

        # Checkpoint: periodico
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                path=os.path.join(CHECKPOINT_DIR_IJEPA(run_path_name), f"checkpoint_epoch_{epoch:03d}.pt"),
                epoch=epoch,
                states={
                    'encoder':        model.encoder.state_dict(),
                    'target_encoder': model.target_encoder.state_dict(),
                    'predictor':      model.predictor.state_dict(),
                },
                optimizer=optimizer,
                scaler=scaler,
                metrics=epoch_metrics,
            )

    # ==========================================================================
    # FINE TRAINING
    # ==========================================================================

    # Checkpoint finale
    save_checkpoint(
        path=os.path.join(CHECKPOINT_DIR_IJEPA(run_path_name), "final_model.pt"),
        epoch=EPOCHS - 1,
        states={
            'encoder':        model.encoder.state_dict(),
            'target_encoder': model.target_encoder.state_dict(),
            'predictor':      model.predictor.state_dict(),
        },
        optimizer=optimizer,
        scaler=scaler,
        metrics=all_results[-1],
    )

    # Summary finale
    save_results({
        'run_name':          run_name,
        'total_epochs':      EPOCHS,
        'best_knn_acc':      best_knn_acc,
        'best_epoch':        best_epoch,
        'final_train_loss':  all_results[-1]['train_loss'],
        'final_val_loss':    all_results[-1]['val_loss'],
        'final_knn_acc':     all_results[-1]['knn_acc'],
        'linear_probe_top1': all_results[-1].get('linear_probe_top1', None),
        'linear_probe_top5': all_results[-1].get('linear_probe_top5', None),
    }, os.path.join(RESULTS_DIR_IJEPA(run_path_name), "summary.json"))

    print("\n" + "=" * 60)
    print(f"Training completato!")
    print(f"  Best k-NN:  {best_knn_acc*100:.2f}% (epoca {best_epoch})")
    print(f"  Checkpoint: {CHECKPOINT_DIR_IJEPA(run_path_name)}")
    print(f"  Risultati:  {RESULTS_DIR_IJEPA(run_path_name)}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()