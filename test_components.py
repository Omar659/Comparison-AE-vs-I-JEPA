"""
test_components.py

Test rapidi per verificare il funzionamento di:
  1. Costanti
  2. PatchEmbed
  3. Positional Embedding 2D
  4. VisionTransformerEncoder (full + masked)
  5. VisionTransformerEncoder (masked)
  6. Predictor
  7. Encoder + Predictor end-to-end
  8. MaskGenerator (singola immagine)
  9. MaskGenerator (batch) + pipeline completa
  10. Dataloader

Esegui con:
    python test_components.py
"""

import os
import tempfile

import torch
import traceback

# =============================================================================
# UTILITY
# =============================================================================

PASS = "  [PASS]"
FAIL = "  [FAIL]"

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(condition: bool, msg: str):
    if condition:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
        raise AssertionError(msg)


# =============================================================================
# 1. COSTANTI
# =============================================================================

section("1. COSTANTI")
try:
    from src.constants import (
        SEED, DEVICE, USE_BFLOAT16,
        DATA_DIR, IMG_SIZE, PATCH_SIZE, NUM_WORKERS, BATCH_SIZE,
        EMBED_DIM, ENCODER_DEPTH, NUM_HEADS, MLP_RATIO, DROP, ATTN_DROP,
        DECODER_EMBED_DIM,
    )
    check(IMG_SIZE % PATCH_SIZE == 0,
          f"IMG_SIZE ({IMG_SIZE}) divisibile per PATCH_SIZE ({PATCH_SIZE})")
    check(EMBED_DIM % NUM_HEADS == 0,
          f"EMBED_DIM ({EMBED_DIM}) divisibile per NUM_HEADS ({NUM_HEADS})")
    check(isinstance(DEVICE, torch.device),
          f"DEVICE è un torch.device: {DEVICE}")
    print(f"  INFO  DEVICE={DEVICE}, USE_BFLOAT16={USE_BFLOAT16}")
except Exception:
    traceback.print_exc()


# =============================================================================
# 2. PATCH EMBED
# =============================================================================

section("2. PATCH EMBED")
try:
    from src.models.vit import PatchEmbed

    B = 2
    pe = PatchEmbed(img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                    in_chans=3, embed_dim=EMBED_DIM)
    x  = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    out = pe(x)

    N_expected = (IMG_SIZE // PATCH_SIZE) ** 2
    check(out.shape == (B, N_expected, EMBED_DIM),
          f"PatchEmbed output shape: {out.shape} == ({B}, {N_expected}, {EMBED_DIM})")
except Exception:
    traceback.print_exc()


# =============================================================================
# 3. POSITIONAL EMBEDDING 2D
# =============================================================================

section("3. POSITIONAL EMBEDDING 2D (sinusoidale)")
try:
    from src.models.vit import build_2d_sincos_pos_embed

    grid_size = IMG_SIZE // PATCH_SIZE
    pos_emb   = build_2d_sincos_pos_embed(EMBED_DIM, grid_size)

    check(pos_emb.shape == (1, grid_size**2, EMBED_DIM),
          f"pos_embed shape: {pos_emb.shape} == (1, {grid_size**2}, {EMBED_DIM})")
    check(not torch.isnan(pos_emb).any(),
          "Nessun NaN nel positional embedding")
    check(not torch.isinf(pos_emb).any(),
          "Nessun Inf nel positional embedding")
except Exception:
    traceback.print_exc()


# =============================================================================
# 4. VIT ENCODER — modalità FULL (EMA encoder / AE)
# =============================================================================

section("4. VIT ENCODER — full (nessun masking)")
try:
    from src.models.vit import VisionTransformerEncoder

    encoder = VisionTransformerEncoder()
    encoder.eval()

    B   = 2
    x   = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    N   = (IMG_SIZE // PATCH_SIZE) ** 2

    with torch.no_grad():
        out = encoder(x, mask_indices=None)

    check(out.shape == (B, N, EMBED_DIM),
          f"Full encoder output shape: {out.shape} == ({B}, {N}, {EMBED_DIM})")
    check(not torch.isnan(out).any(),  "Nessun NaN nell'output")
    check(not torch.isinf(out).any(),  "Nessun Inf nell'output")
except Exception:
    traceback.print_exc()


# =============================================================================
# 5. VIT ENCODER — modalità MASKED (context encoder)
# =============================================================================

section("5. VIT ENCODER — masked (context encoder)")
try:
    from src.models.vit import VisionTransformerEncoder

    encoder = VisionTransformerEncoder()
    encoder.eval()

    B        = 2
    N        = (IMG_SIZE // PATCH_SIZE) ** 2
    # Simuliamo masking aggressivo stile V-JEPA 2: teniamo ~10% delle patch
    N_vis    = max(1, int(N * 0.10))
    x        = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)

    # Indici delle patch visibili: shape (B, N_vis)
    mask_indices = torch.stack([
        torch.randperm(N)[:N_vis] for _ in range(B)
    ])

    with torch.no_grad():
        out = encoder(x, mask_indices=mask_indices)

    check(out.shape == (B, N_vis, EMBED_DIM),
          f"Masked encoder output shape: {out.shape} == ({B}, {N_vis}, {EMBED_DIM})")
    check(not torch.isnan(out).any(), "Nessun NaN nell'output mascherato")

    print(f"  INFO  N_totale={N}, N_visibili={N_vis} "
          f"({N_vis/N*100:.1f}% delle patch)")
except Exception:
    traceback.print_exc()


# =============================================================================
# 6. PREDICTOR
# =============================================================================

section("6. PREDICTOR")
try:
    from src.models.predictor import Predictor
    from src.constants import PRED_EMBED_DIM, PRED_DEPTH, PRED_NUM_HEADS

    B      = 2
    N      = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
    N_vis  = max(1, int(N * 0.10))           # ~10% visibili  (~20)
    N_mask = N - N_vis                       # ~90% mascherati (~176)

    all_ids     = torch.stack([torch.randperm(N) for _ in range(B)])
    visible_ids = all_ids[:, :N_vis]         # (B, N_vis)
    mask_ids    = all_ids[:, N_vis:]         # (B, N_mask)

    # Token visibili come se venissero dall'encoder
    visible_tokens = torch.randn(B, N_vis, EMBED_DIM)

    predictor = Predictor()
    predictor.eval()

    with torch.no_grad():
        pred = predictor(visible_tokens, visible_ids, mask_ids)

    check(pred.shape == (B, N_mask, EMBED_DIM),
          f"Predictor output shape: {pred.shape} == ({B}, {N_mask}, {EMBED_DIM})")
    check(not torch.isnan(pred).any(), "Nessun NaN nelle predizioni")
    check(not torch.isinf(pred).any(), "Nessun Inf nelle predizioni")
    check(predictor.input_proj.in_features   == EMBED_DIM,
          f"input_proj:  {EMBED_DIM} → {PRED_EMBED_DIM}")
    check(predictor.output_proj.out_features == EMBED_DIM,
          f"output_proj: {PRED_EMBED_DIM} → {EMBED_DIM}")

    print(f"  INFO  N_vis={N_vis}, N_mask={N_mask} "
          f"(masking ratio={N_mask/N*100:.1f}%)")
    print(f"  INFO  depth={PRED_DEPTH}, pred_dim={PRED_EMBED_DIM}, "
          f"num_heads={PRED_NUM_HEADS}")
except Exception:
    traceback.print_exc()


# =============================================================================
# 7. ENCODER + PREDICTOR end-to-end (con loss L1)
# =============================================================================

section("7. ENCODER + PREDICTOR end-to-end")
try:
    from src.models.vit import VisionTransformerEncoder
    from src.models.predictor import Predictor
    import torch.nn.functional as F

    B      = 2
    N      = (IMG_SIZE // PATCH_SIZE) ** 2
    N_vis  = max(1, int(N * 0.10))
    N_mask = N - N_vis

    x       = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    all_ids = torch.stack([torch.randperm(N) for _ in range(B)])
    visible_ids = all_ids[:, :N_vis]
    mask_ids    = all_ids[:, N_vis:]

    encoder   = VisionTransformerEncoder()
    predictor = Predictor()
    encoder.eval()
    predictor.eval()

    with torch.no_grad():
        # Context encoder: solo patch visibili
        visible_tokens = encoder(x, mask_indices=visible_ids)

        # EMA encoder: immagine intera → target
        all_tokens = encoder(x, mask_indices=None)

        # Predictor: predice le patch mascherate
        pred = predictor(visible_tokens, visible_ids, mask_ids)

        # Estrai i target corrispondenti alle patch mascherate
        idx_mask = mask_ids.unsqueeze(-1).expand(-1, -1, EMBED_DIM)
        target   = torch.gather(all_tokens, dim=1, index=idx_mask)

        # Loss L1 solo sulle patch mascherate
        loss = F.l1_loss(pred, target)

    check(pred.shape == target.shape,
          f"pred e target stessa shape: {pred.shape}")
    check(not torch.isnan(loss),
          f"Loss L1 non NaN: {loss.item():.4f}")
    check(0 < loss.item() < 100,
          f"Loss L1 ragionevole: {loss.item():.4f}")

    print(f"  INFO  Loss L1 a pesi random: {loss.item():.4f}")
except Exception:
    traceback.print_exc()


# =============================================================================
# 8. MASK GENERATOR — singola immagine
# =============================================================================

section("8. MASK GENERATOR — singola immagine")
try:
    from src.masks.mask_generator import MaskGenerator
    from src.constants import NUM_MASK_BLOCKS, MIN_VISIBLE_PATCHES

    N   = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
    gen = MaskGenerator()

    visible_ids, mask_ids = gen(batch_size=1)
    visible_ids = visible_ids[0]   # (N_vis,)
    mask_ids    = mask_ids[0]      # (N_mask,)

    # 1. Nessun indice fuori range
    check(visible_ids.max() < N and visible_ids.min() >= 0,
          f"visible_ids nel range [0, {N}): min={visible_ids.min()}, max={visible_ids.max()}")
    check(mask_ids.max() < N and mask_ids.min() >= 0,
          f"mask_ids nel range [0, {N}): min={mask_ids.min()}, max={mask_ids.max()}")

    # 2. Nessun indice duplicato
    check(len(visible_ids.unique()) == len(visible_ids),
          f"visible_ids senza duplicati: {len(visible_ids)} unici")
    check(len(mask_ids.unique()) == len(mask_ids),
          f"mask_ids senza duplicati: {len(mask_ids)} unici")

    # 3. Visible e masked sono disgiunti
    vis_set  = set(visible_ids.tolist())
    mask_set = set(mask_ids.tolist())
    check(len(vis_set & mask_set) == 0,
          "visible_ids e mask_ids disgiunti (nessun indice in comune)")

    # 4. Unione = tutte le patch
    check(len(vis_set | mask_set) == N,
          f"visible u masked = {len(vis_set | mask_set)} == {N} (tutte le patch)")

    # 5. Minimo di patch visibili rispettato
    check(len(visible_ids) >= MIN_VISIBLE_PATCHES,
          f"Patch visibili >= MIN_VISIBLE_PATCHES: {len(visible_ids)} >= {MIN_VISIBLE_PATCHES}")

    masking_ratio = len(mask_ids) / N * 100
    print(f"  INFO  N_vis={len(visible_ids)}, N_mask={len(mask_ids)}, "
          f"masking ratio={masking_ratio:.1f}%")
except Exception:
    traceback.print_exc()


# =============================================================================
# 9. MASK GENERATOR — batch + pipeline completa con encoder e predictor
# =============================================================================

section("9. MASK GENERATOR — batch + pipeline completa")
try:
    from src.masks.mask_generator import MaskGenerator
    from src.models.vit import VisionTransformerEncoder
    from src.models.predictor import Predictor
    import torch.nn.functional as F

    B   = 4
    N   = (IMG_SIZE // PATCH_SIZE) ** 2
    gen = MaskGenerator()

    # 1. Genera maschere per il batch
    visible_ids, mask_ids = gen(batch_size=B)   # (B, N_vis), (B, N_mask)

    check(visible_ids.shape[0] == B,
          f"visible_ids batch dim: {visible_ids.shape[0]} == {B}")
    check(mask_ids.shape[0] == B,
          f"mask_ids batch dim: {mask_ids.shape[0]} == {B}")

    N_vis  = visible_ids.shape[1]
    N_mask = mask_ids.shape[1]

    print(f"  INFO  Batch={B}, N_vis={N_vis}, N_mask={N_mask}, "
          f"masking={N_mask/N*100:.1f}%")

    # 2. Pipeline completa: MaskGenerator -> Encoder -> Predictor -> Loss
    x         = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    encoder   = VisionTransformerEncoder()
    predictor = Predictor()
    encoder.eval()
    predictor.eval()

    with torch.no_grad():
        # Context encoder: patch visibili
        visible_tokens = encoder(x, mask_indices=visible_ids)

        # EMA encoder: immagine intera
        all_tokens = encoder(x, mask_indices=None)

        # Predictor
        pred = predictor(visible_tokens, visible_ids, mask_ids)

        # Target: token mascherati dall'EMA encoder
        idx = mask_ids.unsqueeze(-1).expand(-1, -1, EMBED_DIM)
        target = torch.gather(all_tokens, dim=1, index=idx)

        loss = F.l1_loss(pred, target)

    check(pred.shape == target.shape,
          f"pred == target shape: {pred.shape}")
    check(not torch.isnan(loss), f"Loss non NaN: {loss.item():.4f}")
    check(0 < loss.item() < 100, f"Loss ragionevole: {loss.item():.4f}")

    print(f"  INFO  Pipeline completa OK — Loss L1: {loss.item():.4f}")
except Exception:
    traceback.print_exc()


# =============================================================================
# 10. IJEPA — sistema completo
# =============================================================================

section("10. IJEPA — sistema completo")
try:
    from src.models.vit import VisionTransformerEncoder
    from src.models.predictor import Predictor
    from src.models.ijepa import IJEPA
    from src.masks.mask_generator import MaskGenerator

    B   = 2
    N   = (IMG_SIZE // PATCH_SIZE) ** 2
    gen = MaskGenerator()

    encoder   = VisionTransformerEncoder()
    predictor = Predictor()
    model     = IJEPA(encoder, predictor)
    model.eval()

    # 1. Target encoder deve essere frozen
    for param in model.target_encoder.parameters():
        check(not param.requires_grad,
              "target_encoder parametri frozen (requires_grad=False)")
        break  # basta il primo

    # 2. Encoder e predictor devono essere trainabili
    trainable = model.trainable_parameters()
    check(len(trainable) > 0, f"trainable_parameters() non vuoto: {len(trainable)} tensori")

    # 3. Target encoder non deve essere in trainable_parameters
    target_ids = {id(p) for p in model.target_encoder.parameters()}
    trainable_ids = {id(p) for p in trainable}
    check(len(target_ids & trainable_ids) == 0,
          "target_encoder NON è in trainable_parameters()")

    # 4. Forward pass completo
    x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    visible_ids, mask_ids = gen(batch_size=B)

    with torch.no_grad():
        loss, pred = model(x, visible_ids, mask_ids)

    N_mask = mask_ids.shape[1]
    check(pred.shape == (B, N_mask, EMBED_DIM),
          f"pred shape: {pred.shape} == ({B}, {N_mask}, {EMBED_DIM})")
    check(not torch.isnan(loss), f"Loss non NaN: {loss.item():.4f}")
    check(0 < loss.item() < 100, f"Loss ragionevole: {loss.item():.4f}")

    # 5. initialize_target_encoder — i pesi devono essere identici dopo la chiamata
    model.initialize_target_encoder()
    for p_enc, p_tgt in zip(model.encoder.parameters(),
                             model.target_encoder.parameters()):
        check(torch.allclose(p_enc, p_tgt),
              "initialize_target_encoder: encoder e target_encoder hanno pesi identici")
        break  # basta il primo layer

    print(f"  INFO  Loss L1 IJEPA a pesi random: {loss.item():.4f}")
    print(f"  INFO  Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    tgt_params = sum(p.numel() for p in model.target_encoder.parameters())
    print(f"  INFO  Target encoder params (frozen): {tgt_params/1e6:.1f}M")
except Exception:
    traceback.print_exc()


# =============================================================================
# 11. EMA
# =============================================================================

section("11. EMA")
try:
    from src.models.vit import VisionTransformerEncoder
    from src.utils.ema import update_ema, get_momentum
    from src.constants import EMA_MOMENTUM_START, EMA_MOMENTUM_END

    # --- get_momentum ---
    total = 1000

    m_start = get_momentum(0,    total, EMA_MOMENTUM_START, EMA_MOMENTUM_END)
    m_mid   = get_momentum(500,  total, EMA_MOMENTUM_START, EMA_MOMENTUM_END)
    m_end   = get_momentum(1000, total, EMA_MOMENTUM_START, EMA_MOMENTUM_END)

    check(abs(m_start - EMA_MOMENTUM_START) < 1e-6,
          f"get_momentum step=0: {m_start:.6f} == {EMA_MOMENTUM_START}")
    check(abs(m_end - EMA_MOMENTUM_END) < 1e-6,
          f"get_momentum step=total: {m_end:.6f} == {EMA_MOMENTUM_END}")
    check(EMA_MOMENTUM_START <= m_mid <= EMA_MOMENTUM_END,
          f"get_momentum step=mid: {m_mid:.6f} in [{EMA_MOMENTUM_START}, {EMA_MOMENTUM_END}]")

    # schedule fisso (start == end)
    m_fixed = get_momentum(500, total, 0.996, 0.996)
    check(abs(m_fixed - 0.996) < 1e-6,
          f"get_momentum fisso: {m_fixed:.6f} == 0.996")

    # --- update_ema ---
    enc = VisionTransformerEncoder()
    tgt = VisionTransformerEncoder()

    # Rendi i pesi diversi per testare l'aggiornamento
    with torch.no_grad():
        for p in tgt.parameters():
            p.fill_(0.0)

    # Con momentum=1.0 il target non deve cambiare
    update_ema(enc, tgt, momentum=1.0)
    for p_tgt in tgt.parameters():
        check(p_tgt.data.abs().max() < 1e-6,
              "update_ema momentum=1.0: target invariato")
        break

    # Con momentum=0.0 il target deve diventare uguale all'encoder
    update_ema(enc, tgt, momentum=0.0)
    for p_enc, p_tgt in zip(enc.parameters(), tgt.parameters()):
        check(torch.allclose(p_enc, p_tgt),
              "update_ema momentum=0.0: target == encoder")
        break

    # Con momentum=0.5 il target deve essere la media
    with torch.no_grad():
        for p in tgt.parameters():
            p.fill_(0.0)
    update_ema(enc, tgt, momentum=0.5)
    for p_enc, p_tgt in zip(enc.parameters(), tgt.parameters()):
        expected = 0.5 * p_enc.data   # 0.5 * enc + 0.5 * 0.0
        check(torch.allclose(p_tgt, expected, atol=1e-5),
              "update_ema momentum=0.5: target == 0.5 * encoder")
        break

    # Il target encoder non deve avere gradienti dopo update
    for p_tgt in tgt.parameters():
        check(p_tgt.grad is None,
              "update_ema: nessun gradiente nel target encoder")
        break

    print(f"  INFO  Schedule: {EMA_MOMENTUM_START} → {m_mid:.4f} → {EMA_MOMENTUM_END}")
except Exception:
    traceback.print_exc()


# =============================================================================
# 12. K-NN ACCURACY
# =============================================================================

section("12. K-NN ACCURACY")
try:
    from src.models.vit import VisionTransformerEncoder
    from src.utils.knn import extract_features, knn_accuracy
    from torch.utils.data import DataLoader, TensorDataset

    # Dataset sintetico: 200 train, 50 test, 10 classi, img 224x224
    torch.manual_seed(SEED)
    N_TRAIN, N_TEST, N_CLASSES = 200, 50, 10
    B_SIZE = 32

    # Immagini e label sintetiche
    train_imgs   = torch.randn(N_TRAIN, 3, IMG_SIZE, IMG_SIZE)
    train_labels = torch.randint(0, N_CLASSES, (N_TRAIN,))
    test_imgs    = torch.randn(N_TEST, 3, IMG_SIZE, IMG_SIZE)
    test_labels  = torch.randint(0, N_CLASSES, (N_TEST,))

    train_loader_knn = DataLoader(TensorDataset(train_imgs, train_labels), batch_size=B_SIZE)
    test_loader_knn  = DataLoader(TensorDataset(test_imgs,  test_labels),  batch_size=B_SIZE)

    encoder = VisionTransformerEncoder()

    # Test extract_features
    feats, labels = extract_features(encoder, train_loader_knn, torch.device('cpu'))
    check(feats.shape == (N_TRAIN, EMBED_DIM),
          f"extract_features shape: {feats.shape} == ({N_TRAIN}, {EMBED_DIM})")
    check(labels.shape == (N_TRAIN,),
          f"extract_features labels: {labels.shape}")

    # Verifica L2 normalizzazione
    norms = feats.norm(dim=-1)
    check(torch.allclose(norms, torch.ones(N_TRAIN), atol=1e-5),
          f"Features L2-normalizzate (norms ~1): min={norms.min():.4f}, max={norms.max():.4f}")

    # Test knn_accuracy — con pesi random su dati random ci aspettiamo ~1/N_CLASSES
    from src.constants import KNN_K, KNN_TEMPERATURE
    acc = knn_accuracy(encoder, train_loader_knn, test_loader_knn, torch.device('cpu'), k=KNN_K)
    check(0.0 <= acc <= 1.0, f"k-NN accuracy in [0,1]: {acc:.4f}")

    print(f"  INFO  k-NN accuracy (pesi random, dati random): {acc:.4f}")
    print(f"        Atteso ~{1/N_CLASSES:.4f} (chance level)")
except Exception:
    traceback.print_exc()


# =============================================================================
# 13. LINEAR PROBE
# =============================================================================

section("13. LINEAR PROBE")
try:
    from src.models.vit import VisionTransformerEncoder
    from src.utils.linear_probe import linear_probe
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)
    N_TRAIN, N_TEST, N_CLASSES = 200, 50, 10
    B_SIZE = 32

    train_imgs   = torch.randn(N_TRAIN, 3, IMG_SIZE, IMG_SIZE)
    train_labels = torch.randint(0, N_CLASSES, (N_TRAIN,))
    test_imgs    = torch.randn(N_TEST, 3, IMG_SIZE, IMG_SIZE)
    test_labels  = torch.randint(0, N_CLASSES, (N_TEST,))

    train_loader_lp = DataLoader(TensorDataset(train_imgs, train_labels), batch_size=B_SIZE)
    test_loader_lp  = DataLoader(TensorDataset(test_imgs,  test_labels),  batch_size=B_SIZE)

    encoder = VisionTransformerEncoder()

    # Verifica che l'encoder rimanga frozen durante la probe
    params_before = [p.data.clone() for p in encoder.parameters()]

    results = linear_probe(
        encoder, train_loader_lp, test_loader_lp,
        device=torch.device('cpu'),
        num_classes=N_CLASSES,
        epochs=3,    # poche epoche per il test (override del default per velocità)
    )

    params_after = [p.data.clone() for p in encoder.parameters()]
    encoder_unchanged = all(
        torch.allclose(b, a) for b, a in zip(params_before, params_after)
    )
    check(encoder_unchanged,
          "Encoder rimasto frozen durante la linear probe")

    check('top1' in results and 'top5' in results,
          f"Risultati contengono top1 e top5: {results}")
    check(0.0 <= results['top1'] <= 1.0,
          f"top-1 accuracy in [0,1]: {results['top1']:.4f}")
    check(0.0 <= results['top5'] <= 1.0,
          f"top-5 accuracy in [0,1]: {results['top5']:.4f}")
    check(results['top5'] >= results['top1'],
          f"top-5 >= top-1: {results['top5']:.4f} >= {results['top1']:.4f}")

    print(f"  INFO  Linear probe (pesi random, dati random, 3 epoche):")
    print(f"        top-1: {results['top1']:.4f}  top-5: {results['top5']:.4f}")
except Exception:
    traceback.print_exc()


# =============================================================================
# 14. TRAIN UTILITIES (get_lr, get_weight_decay, checkpoint, results)
# =============================================================================

section("14. TRAIN UTILITIES")
try:
    from src.utils.training import get_lr, get_weight_decay, save_checkpoint, load_checkpoint, save_results
    from src.constants import CHECKPOINT_DIR_IJEPA, RESULTS_DIR_IJEPA

    # --- CHECKPOINT_DIR_IJEPA e RESULTS_DIR_IJEPA come lambda ---
    check(CHECKPOINT_DIR_IJEPA() == "./checkpoints/ijepa",
          f"CHECKPOINT_DIR_IJEPA() base: {CHECKPOINT_DIR_IJEPA()}")
    check(CHECKPOINT_DIR_IJEPA("0") == "./checkpoints/ijepa/0",
          f"CHECKPOINT_DIR_IJEPA('0'): {CHECKPOINT_DIR_IJEPA('0')}")
    check(RESULTS_DIR_IJEPA() == "./results/ijepa",
          f"RESULTS_DIR_IJEPA() base: {RESULTS_DIR_IJEPA()}")
    check(RESULTS_DIR_IJEPA("42") == "./results/ijepa/42",
          f"RESULTS_DIR_IJEPA('42'): {RESULTS_DIR_IJEPA('42')}")

    # --- Logica run_path_name (stessa usata in train_ijepa.py) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        import unittest.mock as mock

        # Simula RESULTS_DIR_IJEPA che punta a tmpdir
        fake_results_dir = lambda x=None: tmpdir if x is None else os.path.join(tmpdir, x)

        # Caso 1: nessuna run esistente → run_path_name deve diventare "0"
        listdir = os.listdir(fake_results_dir())
        run_path_name = "-1"
        if listdir != []:
            listdir = [int(x) for x in listdir]
            listdir.sort()
            run_path_name = str(listdir[-1])
        new_run = str(int(run_path_name) + 1)
        check(new_run == "0", f"Prima run: run_path_name=0, got {new_run}")

        # Caso 2: esiste già la run "0" → deve diventare "1"
        os.makedirs(os.path.join(tmpdir, "0"))
        listdir = os.listdir(fake_results_dir())
        run_path_name = "-1"
        if listdir != []:
            listdir = [int(x) for x in listdir]
            listdir.sort()
            run_path_name = str(listdir[-1])
        new_run = str(int(run_path_name) + 1)
        check(new_run == "1", f"Seconda run: run_path_name=1, got {new_run}")

        # Caso 3: esistono "0", "1", "5" → deve diventare "6"
        os.makedirs(os.path.join(tmpdir, "1"))
        os.makedirs(os.path.join(tmpdir, "5"))
        listdir = os.listdir(fake_results_dir())
        run_path_name = "-1"
        if listdir != []:
            listdir = [int(x) for x in listdir]
            listdir.sort()
            run_path_name = str(listdir[-1])
        new_run = str(int(run_path_name) + 1)
        check(new_run == "6", f"Run dopo gap: run_path_name=6, got {new_run}")

    print("  INFO  run_path_name logic OK")
    from src.models.vit import VisionTransformerEncoder
    from src.models.predictor import Predictor
    from src.models.ijepa import IJEPA
    from src.constants import (
        BASE_LR, MIN_LR, WARMUP_EPOCHS, EPOCHS,
        WEIGHT_DECAY_START, WEIGHT_DECAY_END,
    )
    import tempfile, os

    # --- get_lr ---
    lr_epoch0    = get_lr(0,          WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)
    lr_warmup    = get_lr(WARMUP_EPOCHS - 1, WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)
    lr_midway    = get_lr(EPOCHS // 2, WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)
    lr_final     = get_lr(EPOCHS - 1,  WARMUP_EPOCHS, EPOCHS, BASE_LR, MIN_LR)

    check(lr_epoch0 < BASE_LR,
          f"LR epoch=0 < BASE_LR: {lr_epoch0:.2e} < {BASE_LR:.2e}  (warmup)")
    check(abs(lr_warmup - BASE_LR) < 1e-8,
          f"LR fine warmup ≈ BASE_LR: {lr_warmup:.2e} ≈ {BASE_LR:.2e}")
    check(MIN_LR <= lr_midway <= BASE_LR,
          f"LR metà training nel range: {lr_midway:.2e}")
    check(lr_final >= MIN_LR,
          f"LR finale >= MIN_LR: {lr_final:.2e} >= {MIN_LR:.2e}")
    check(lr_midway < lr_warmup,
          f"LR decresce dopo warmup: {lr_midway:.2e} < {lr_warmup:.2e}")

    print(f"  INFO  LR schedule: {lr_epoch0:.2e} → {BASE_LR:.2e} → {lr_final:.2e}")

    # --- get_weight_decay ---
    wd_start = get_weight_decay(0,          EPOCHS, WEIGHT_DECAY_START, WEIGHT_DECAY_END)
    wd_end   = get_weight_decay(EPOCHS - 1, EPOCHS, WEIGHT_DECAY_START, WEIGHT_DECAY_END)
    wd_mid   = get_weight_decay(EPOCHS // 2, EPOCHS, WEIGHT_DECAY_START, WEIGHT_DECAY_END)

    check(abs(wd_start - WEIGHT_DECAY_START) < 1e-6,
          f"WD epoch=0 == WEIGHT_DECAY_START: {wd_start:.4f}")
    check(abs(wd_end - WEIGHT_DECAY_END) < 1e-6,
          f"WD fine training == WEIGHT_DECAY_END: {wd_end:.4f}")
    check(WEIGHT_DECAY_START <= wd_mid <= WEIGHT_DECAY_END,
          f"WD metà training nel range: {wd_mid:.4f}")

    print(f"  INFO  WD schedule: {wd_start:.4f} → {wd_mid:.4f} → {wd_end:.4f}")

    # --- save_results / JSON ---
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "test_results.json")
        fake_results = [
            {'epoch': 0, 'train_loss': 1.5, 'val_loss': 1.6, 'knn_acc': 0.10},
            {'epoch': 1, 'train_loss': 1.3, 'val_loss': 1.4, 'knn_acc': 0.12},
        ]
        save_results(fake_results, results_path)
        check(os.path.exists(results_path), "results.json creato")

        import json
        with open(results_path) as f:
            loaded = json.load(f)
        check(len(loaded) == 2, f"results.json contiene {len(loaded)} epoche")
        check(loaded[1]['knn_acc'] == 0.12, "Dati corretti nel JSON")

    # --- save_checkpoint / load_checkpoint ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")

        encoder   = VisionTransformerEncoder()
        predictor = Predictor()
        model     = IJEPA(encoder, predictor)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        scaler    = torch.amp.GradScaler('cuda', enabled=False)
        metrics   = {'epoch': 5, 'train_loss': 0.9, 'knn_acc': 0.25}

        save_checkpoint(
            path=ckpt_path,
            epoch=5,
            states={
                'encoder':        model.encoder.state_dict(),
                'target_encoder': model.target_encoder.state_dict(),
                'predictor':      model.predictor.state_dict(),
            },
            optimizer=optimizer,
            scaler=scaler,
            metrics=metrics,
        )
        check(os.path.exists(ckpt_path), "Checkpoint salvato")

        # Modifica i pesi per verificare che il load li ripristini
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.fill_(0.0)
                break

        ckpt_loaded = load_checkpoint(ckpt_path, torch.device('cpu'))
        check(ckpt_loaded['epoch'] == 5,
              f"Epoch corretta dopo load: {ckpt_loaded['epoch']}")
        check(ckpt_loaded['metrics']['knn_acc'] == 0.25,
              f"Metrics corrette dopo load: {ckpt_loaded['metrics']['knn_acc']}")
        check('encoder' in ckpt_loaded['states'],
              "states contiene 'encoder'")
        check('target_encoder' in ckpt_loaded['states'],
              "states contiene 'target_encoder'")
        check('predictor' in ckpt_loaded['states'],
              "states contiene 'predictor'")

        # Carica i pesi e verifica che siano stati ripristinati
        model.encoder.load_state_dict(ckpt_loaded['states']['encoder'])
        for p in model.encoder.parameters():
            check(p.abs().max() > 0,
                  "Pesi encoder ripristinati dopo load_checkpoint")
            break

    print("  INFO  Checkpoint save/load OK")
except Exception:
    traceback.print_exc()


# =============================================================================
# 15. DATALOADER
# =============================================================================

section("15. DATALOADER")
try:
    import os
    from src.dataset import get_dataloaders

    if not os.path.isdir(DATA_DIR):
        print(f"  SKIP  DATA_DIR non trovata ({DATA_DIR}), skip dataloader test.")
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            DATA_DIR,
            batch_size=4,       # piccolo per il test
            num_workers=0,      # 0 worker per semplicità in test
        )

        # --- Train ---
        imgs, labels = next(iter(train_loader))
        check(imgs.shape == (4, 3, IMG_SIZE, IMG_SIZE),
              f"Train batch shape: {imgs.shape}")
        check(imgs.dtype == torch.float32,
              f"Dtype immagini train: {imgs.dtype}")
        check(imgs.min() > -5 and imgs.max() < 5,
              f"Range valori train: [{imgs.min():.2f}, {imgs.max():.2f}]")

        # --- Val ---
        imgs_v, labels_v = next(iter(val_loader))
        check(imgs_v.shape[1:] == (3, IMG_SIZE, IMG_SIZE),
              f"Val batch shape: {imgs_v.shape}")

        # --- Test ---
        imgs_t, labels_t = next(iter(test_loader))
        check(imgs_t.shape[1:] == (3, IMG_SIZE, IMG_SIZE),
              f"Test batch shape: {imgs_t.shape}")

        # Test e val devono avere la stessa dimensione totale
        check(len(test_loader.dataset) == len(val_loader.dataset),
              f"Test size == Val size: {len(test_loader.dataset)} == {len(val_loader.dataset)}")

        # Train e test devono essere disgiunti
        train_indices = set(train_loader.dataset.indices)
        test_indices  = set(test_loader.dataset.indices)
        check(len(train_indices & test_indices) == 0,
              "Train e Test sono disgiunti (nessun indice in comune)")

        print(f"  INFO  Train={len(train_loader.dataset)}, "
              f"Val={len(val_loader.dataset)}, "
              f"Test={len(test_loader.dataset)} campioni")
        print(f"  INFO  Batches — Train: {len(train_loader)}, "
              f"Val: {len(val_loader)}, Test: {len(test_loader)}")
except Exception:
    traceback.print_exc()




# =============================================================================
# 11. VISUALIZZAZIONE MASCHERA
# =============================================================================

section("11. VISUALIZZAZIONE MASCHERA")
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from src.masks.mask_generator import MaskGenerator

    def visualize_mask(save_path: str = "mask_visualization.png"):
        """
        Crea e salva una visualizzazione della maschera su un'immagine random.

        Mostra tre pannelli affiancati:
          - Immagine originale (rumore colorato per simulare una vera immagine)
          - Patch visibili (in colore) e patch mascherate (in grigio scuro)
          - Griglia con colori: verde = visibile, rosso = mascherato
        """
        N        = (IMG_SIZE // PATCH_SIZE) ** 2
        grid_h   = IMG_SIZE // PATCH_SIZE   # 14
        grid_w   = IMG_SIZE // PATCH_SIZE   # 14

        # --- Immagine sintetica colorata (in [0,1]) ---
        torch.manual_seed(SEED)
        img = torch.rand(3, IMG_SIZE, IMG_SIZE)   # (C, H, W)

        # --- Genera maschera ---
        gen = MaskGenerator()
        visible_ids, mask_ids = gen(batch_size=1)
        visible_ids = visible_ids[0]   # (N_vis,)
        mask_ids    = mask_ids[0]      # (N_mask,)

        # Costruisce una mappa booleana 2D: True = mascherato
        is_masked = torch.zeros(N, dtype=torch.bool)
        is_masked[mask_ids] = True
        is_masked_2d = is_masked.reshape(grid_h, grid_w)   # (14, 14)

        # --- Costruisce immagine con patch mascherate oscurate ---
        img_masked = img.clone()
        for idx in mask_ids.tolist():
            r = idx // grid_w
            c = idx % grid_w
            pr, pc = r * PATCH_SIZE, c * PATCH_SIZE
            img_masked[:, pr:pr+PATCH_SIZE, pc:pc+PATCH_SIZE] = 0.15  # grigio scuro

        # Converti in HWC numpy per matplotlib
        img_np        = img.permute(1, 2, 0).numpy()
        img_masked_np = img_masked.permute(1, 2, 0).numpy()

        # --- Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Mask visualization — "
            f"visible: {len(visible_ids)} patch ({len(visible_ids)/N*100:.1f}%)  |  "
            f"masked: {len(mask_ids)} patch ({len(mask_ids)/N*100:.1f}%)",
            fontsize=13, fontweight='bold'
        )

        # Pannello 1: immagine originale
        axes[0].imshow(img_np)
        axes[0].set_title("Immagine originale")
        axes[0].axis('off')

        # Pannello 2: immagine con patch mascherate
        axes[1].imshow(img_masked_np)
        axes[1].set_title("Input context encoder (patch mascherate in grigio)")
        axes[1].axis('off')

        # Pannello 3: griglia colorata (verde=visibile, rosso=mascherato)
        color_grid = np.zeros((grid_h, grid_w, 3))
        color_grid[is_masked_2d.numpy()]  = [0.85, 0.20, 0.20]   # rosso  = mascherato
        color_grid[~is_masked_2d.numpy()] = [0.20, 0.75, 0.35]   # verde  = visibile

        axes[2].imshow(color_grid, interpolation='nearest')
        axes[2].set_title(f"Griglia patch ({grid_h}x{grid_w})")
        axes[2].set_xticks(range(grid_w))
        axes[2].set_yticks(range(grid_h))
        axes[2].tick_params(labelsize=6)
        axes[2].grid(True, color='white', linewidth=0.5)

        # Legenda
        legend = [
            mpatches.Patch(color=[0.20, 0.75, 0.35], label=f'Visibile ({len(visible_ids)})'),
            mpatches.Patch(color=[0.85, 0.20, 0.20], label=f'Mascherato ({len(mask_ids)})'),
        ]
        axes[2].legend(handles=legend, loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  INFO  Visualizzazione salvata in: {os.path.abspath(save_path)}")

    visualize_mask("mask_visualization.png")
    check(os.path.exists("mask_visualization.png"),
          "File mask_visualization.png creato")

except ImportError:
    print("  SKIP  matplotlib non installato — pip install matplotlib")
except Exception:
    traceback.print_exc()


# =============================================================================
# 17. DECODER E PATCHING UTILS
# =============================================================================

section("17. DECODER E PATCHING UTILS")
try:
    from src.models.decoder import VisionTransformerDecoder
    from src.utils.patching import patchify, unpatchify
    from src.constants import DECODER_EMBED_DIM
    
    B = 2
    N = (IMG_SIZE // PATCH_SIZE) ** 2
    
    # --- Test Patching Utilities ---
    imgs_original = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    patches = patchify(imgs_original, PATCH_SIZE)
    imgs_reconstructed = unpatchify(patches, PATCH_SIZE, 3)
    
    check(patches.shape == (B, N, PATCH_SIZE**2 * 3),
          f"Patchify shape: {patches.shape}")
    check(torch.allclose(imgs_original, imgs_reconstructed, atol=1e-6),
          "Unpatchify è l'esatta inversa di Patchify")

    # --- Test Decoder ---
    decoder = VisionTransformerDecoder()
    decoder.eval()
    
    # TEST 1: Full Sequence (es. Phase 2 JEPA senza masking)
    x_full = torch.randn(B, N, EMBED_DIM)
    with torch.no_grad():
        out_full = decoder(x_full, ids_restore=None)
        
    check(out_full.shape == (B, N, PATCH_SIZE**2 * 3),
          f"Decoder (full) output shape: {out_full.shape}")
          
    # TEST 2: Masked Sequence (es. VAE)
    N_vis = max(1, int(N * 0.25)) # Simuliamo un 75% di masking
    x_masked = torch.randn(B, N_vis, EMBED_DIM)
    
    # Creiamo un finto ids_restore mescolando gli indici
    ids_restore = torch.stack([torch.randperm(N) for _ in range(B)])
    
    with torch.no_grad():
        out_masked = decoder(x_masked, ids_restore=ids_restore)
        
    check(out_masked.shape == (B, N, PATCH_SIZE**2 * 3),
          f"Decoder (masked) output shape: {out_masked.shape}")

except Exception:
    traceback.print_exc()

# =============================================================================
# 18. VARIATIONAL MASKED AUTOENCODER (VAE)
# =============================================================================

section("18. VARIATIONAL MASKED AUTOENCODER (VAE)")
try:
    from src.models.vit import VisionTransformerEncoder
    from src.models.decoder import VisionTransformerDecoder
    from src.models.vae import VariationalMaskedAE
    from src.utils.patching import patchify
    from src.masks.mask_generator import MaskGenerator
    from src.constants import VAE_LATENT_DIM
    import torch.nn.functional as F

    B = 4
    N = (IMG_SIZE // PATCH_SIZE) ** 2
    
    # 1. Inizializza i sottomoduli
    encoder = VisionTransformerEncoder()
    # IMPORTANTE: Il decoder deve aspettarsi in input VAE_LATENT_DIM (512)
    decoder = VisionTransformerDecoder(embed_dim=VAE_LATENT_DIM)
    
    # 2. Inizializza il VAE
    vae = VariationalMaskedAE(encoder, decoder)
    vae.train() # Mettiamo in train per testare il reparameterize (aggiunta del rumore)
    
    # 3. Genera dati finti e maschere con il MaskGenerator
    x = torch.randn(B, 3, IMG_SIZE, IMG_SIZE)
    gen = MaskGenerator()
    visible_ids, mask_ids = gen(batch_size=B)
    N_vis = visible_ids.shape[1]
    
    # 4. Forward
    pred_patches, mu, logvar = vae(x, visible_ids, mask_ids)
    
    # Check Shapes
    check(pred_patches.shape == (B, N, PATCH_SIZE**2 * 3), 
          f"Output Decoder shape: {pred_patches.shape}")
    check(mu.shape == (B, N_vis, VAE_LATENT_DIM), 
          f"Mu shape: {mu.shape}")
    check(logvar.shape == (B, N_vis, VAE_LATENT_DIM), 
          f"Logvar shape: {logvar.shape}")
          
    # 5. Simulazione calcolo Loss
    target_patches = patchify(x, PATCH_SIZE)
    
    # Loss di Ricostruzione (MSE su tutte le patch)
    recon_loss = F.mse_loss(pred_patches, target_patches)
    
    # KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Calcolata solo sui token visibili
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    loss = recon_loss + 0.00025 * kl_loss
    
    check(not torch.isnan(loss), f"Loss VAE calcolabile e non-NaN: {loss.item():.4f}")
    print(f"  INFO  Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")

except Exception:
    traceback.print_exc()

# =============================================================================
# RIEPILOGO
# =============================================================================

print(f"\n{'='*60}")
print("  TEST COMPLETATI")
print(f"{'='*60}\n")