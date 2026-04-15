import random
import torch
import numpy as np

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# General Settings
NUM_FEATURE = 32
NUM_CLASSES = 3
PACKET_NUM = 20
CLIENT_LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 64

# DataLoader Settings for Apple Silicon
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on Mac

# Gated AdaBoost Settings
GATING_K = 3 # Top-k learners to select
GATING_TAU = 1.0 # Temperature for softmax
GATING_LR = 1e-3
GATING_WEIGHT_DECAY = 1e-2
GATING_EPOCHS = 50
GATING_VAL_RATIO = 0.1 # 10% of training data for validation
GATING_PATIENCE = 10 # epochs to wait before early stopping
GATING_MIN_DELTA = 0.001 # minimum change to qualify as improvement
GATING_GRAD_CLIP = 1.0 # gradient clipping
GATING_LAMBDA_LB = 0.01 # load balancing loss weight
LAMBDA_KL = 1.0
LAMBDA_RANK = 0.1
LAMBDA_SPARSE = 0.1
LAMBDA_MARGIN = 1.0

# Device Configuration for MacBook M-series and others
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Data Paths
DATA_DIR = "Data"
SOURCE_PATH = f"{DATA_DIR}/Domain 1_32.feather"
TARGET_PATH = f"{DATA_DIR}/Domain 2_32.feather"
TARGET_TEST_RATIO = 0.95  # 95% of target domain for testing (only 5% for training - limited target data scenario)
