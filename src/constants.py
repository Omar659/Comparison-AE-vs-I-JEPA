import torch

# =============================================================================
# GLOBAL & HARDWARE
# =============================================================================
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BFLOAT16 = True  # RTX 4090 supporta bfloat16 nativamente

# =============================================================================
# DATASET
# =============================================================================
DATA_DIR    = "./data/ImageNet100"
IMG_SIZE    = 224
CHANNELS    = 3
PATCH_SIZE  = 16
NUM_WORKERS = 8
BATCH_SIZE  = 128

# =============================================================================
# MODEL ARCHITECTURE (ViT-Base)
# =============================================================================
EMBED_DIM     = 768
ENCODER_DEPTH = 12
NUM_HEADS     = 12
MLP_RATIO     = 4.0
DROP          = 0.0
ATTN_DROP     = 0.0

# =============================================================================
# PREDICTOR (narrow ViT)
# =============================================================================
PRED_EMBED_DIM = 384   # più stretto dell'encoder (768)
PRED_DEPTH     = 6     # per ViT-Base; 12 per ViT-L/H
PRED_NUM_HEADS = 6     # PRED_EMBED_DIM // 64

# =============================================================================
# MASKING (stile V-JEPA 2 adattato alle immagini)
# =============================================================================
# Il context encoder vede solo le patch VISIBILI (~10-15%)
# L'EMA encoder vede l'immagine intera (100%)

# Numero di blocchi da mascherare per immagine
# V-JEPA usa 8 blocchi short-range + 2 long-range -> union ~90% mascherato
NUM_MASK_BLOCKS     = 8       # blocchi da campionare e unire

# Scala spaziale di ogni blocco rispetto all'immagine
# (fraction dell'area totale coperta da un singolo blocco)
MASK_BLOCK_SCALE    = (0.10, 0.20)   # ogni blocco copre 10-20% delle patch

# Aspect ratio dei blocchi (altezza/larghezza)
MASK_ASPECT_RATIO   = (0.75, 1.50)

# Numero minimo di patch visibili garantite al context encoder
MIN_VISIBLE_PATCHES = 10

# =============================================================================
# EMA (Exponential Moving Average - target encoder)
# =============================================================================
EMA_MOMENTUM_START = 0.996   # momentum iniziale (basso = aggiornamento veloce)
EMA_MOMENTUM_END   = 1.0     # momentum finale   (1.0 = target encoder frozen)

# =============================================================================
# K-NN EVALUATION
# =============================================================================
KNN_K           = 20      # numero di vicini - standard in letteratura SSL
KNN_TEMPERATURE = 0.07    # temperatura per weighted k-NN (SimCLR/DINO default)

# =============================================================================
# LINEAR PROBE EVALUATION
# =============================================================================
LINEAR_PROBE_EPOCHS = 20     # epoche di training del classificatore lineare
LINEAR_PROBE_LR     = 1e-3   # learning rate del classificatore
LINEAR_PROBE_EVERY  = 10     # ogni quante epoche di pretraining fare la probe

# =============================================================================
# TRAINING (I-JEPA)
# =============================================================================
EPOCHS        = 100
WARMUP_EPOCHS = 15       # epoche warmup lineare (da I-JEPA paper)

# Learning Rate
BASE_LR       = 1e-3     # LR massimo dopo warmup (I-JEPA paper)
MIN_LR        = 1e-6     # LR minimo a fine cosine decay

# Weight Decay - aumentato linearmente durante il training (I-JEPA paper)
WEIGHT_DECAY_START = 0.04
WEIGHT_DECAY_END   = 0.40

# Gradient clipping
GRAD_CLIP = 1.0

# Numero di classi del dataset
NUM_CLASSES = 100   # ImageNet-100

# =============================================================================
# CHECKPOINT & LOGGING
# =============================================================================
CHECKPOINT_DIR_IJEPA   = lambda x=None : "./checkpoints/ijepa" + ("/" + x if x else "")
CHECKPOINT_EVERY = 10     # ogni quante epoche salvare un checkpoint

RESULTS_DIR_IJEPA      = lambda x=None : "./results/ijepa" + ("/" + x if x else "")   # JSON con metriche per analisi a posteriori

WANDB_PROJECT    = "image-jepa"
WANDB_RUN_NAME   = None   # None = generato automaticamente da W&B\

# =============================================================================
# DECODER ARCHITECTURE
# =============================================================================
# Seguendo le specifiche standard di MAE per ViT-Base
DECODER_EMBED_DIM = 512
DECODER_DEPTH     = 8
DECODER_NUM_HEADS = 16

# =============================================================================
# VAE SPECIFIC PARAMETERS
# =============================================================================
VAE_LATENT_DIM = 512        # Dimensione dello spazio latente variazionale (z)
KLD_WEIGHT     = 0.00025    # Peso iniziale per la Kullback-Leibler divergence
CHECKPOINT_DIR_VAE   = lambda x=None : "./checkpoints/vae" + ("/" + x if x else "")
RESULTS_DIR_VAE      = lambda x=None : "./results/vae" + ("/" + x if x else "")

# Parametri di Masking
MASK_RATIO = 0.75  # 75% dell'immagine viene nascosta