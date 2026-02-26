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
NUM_MASK_BLOCKS     = 8
MASK_BLOCK_SCALE    = (0.10, 0.20)
MASK_ASPECT_RATIO   = (0.75, 1.50)
MIN_VISIBLE_PATCHES = 10

# =============================================================================
# EMA (Exponential Moving Average - target encoder)
# =============================================================================
EMA_MOMENTUM_START = 0.996
EMA_MOMENTUM_END   = 1.0

# =============================================================================
# K-NN EVALUATION
# =============================================================================
KNN_K           = 20
KNN_TEMPERATURE = 0.07

# =============================================================================
# LINEAR PROBE EVALUATION
# =============================================================================
LINEAR_PROBE_EPOCHS = 20
LINEAR_PROBE_LR     = 1e-3
LINEAR_PROBE_EVERY  = 10

# =============================================================================
# TRAINING (I-JEPA)
# =============================================================================
EPOCHS        = 100
WARMUP_EPOCHS = 15
BASE_LR       = 1e-3
MIN_LR        = 1e-6
WEIGHT_DECAY_START = 0.04
WEIGHT_DECAY_END   = 0.40
GRAD_CLIP = 1.0
NUM_CLASSES = 100   # ImageNet-100

# =============================================================================
# CHECKPOINT & LOGGING (I-JEPA)
# =============================================================================
CHECKPOINT_DIR_IJEPA   = lambda x=None : "./checkpoints/ijepa" + ("/" + x if x else "")
CHECKPOINT_EVERY       = 10
RESULTS_DIR_IJEPA      = lambda x=None : "./results/ijepa" + ("/" + x if x else "")
WANDB_PROJECT          = "image-jepa"
WANDB_RUN_NAME         = None

# =============================================================================
# DECODER ARCHITECTURE
# =============================================================================
DECODER_EMBED_DIM = 512
DECODER_DEPTH     = 8
DECODER_NUM_HEADS = 16

# =============================================================================
# VAE SPECIFIC PARAMETERS
# =============================================================================
VAE_LATENT_DIM       = 512
KLD_WEIGHT           = 0.00025
CHECKPOINT_DIR_VAE   = lambda x=None : "./checkpoints/vae" + ("/" + x if x else "")
RESULTS_DIR_VAE      = lambda x=None : "./results/vae" + ("/" + x if x else "")

# =============================================================================
# VAE MASKING
# =============================================================================
VAE_MASK_RATIO = 0.75   # stile MAE: nasconde il 75% delle patch