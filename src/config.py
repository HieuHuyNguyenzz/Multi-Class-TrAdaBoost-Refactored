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
NUM_FEATURE = 256
NUM_CLASSES = 3
PACKET_NUM = 20
CLIENT_LR = 1e-3
NUM_EPOCHS = 30
BATCH_SIZE = 64
NUM_ESTIMATORS = 10  # Number of weak learners in ensemble

# DataLoader Settings for Apple Silicon
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on Mac

# Gated AdaBoost Settings
GATING_K = 3 # Top-k learners to select
GATING_TAU = 1.0 # Temperature for softmax
GATING_LR = 1e-3
GATING_WEIGHT_DECAY = 1e-2
GATING_EPOCHS = 30
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
SOURCE_PATH = f"{DATA_DIR}/Domain 1_256.feather"
TARGET_PATH = f"{DATA_DIR}/Domain 2_256.feather"
TARGET_TEST_RATIO = 0.90  # 90% of target domain for testing
TARGET_TRAIN_LABELED_RATIO = 0.2  # 20% of the training 10% is labeled, 80% is unlabeled
