import torch

# General Settings
NUM_FEATURE = 32  # Updated for new data format (32 features instead of 256)
NUM_CLASSES = 3
PACKET_NUM = 20
CLIENT_LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 64  # Increased for better GPU utilization on M-series
SEED = 42

# DataLoader Settings for Apple Silicon
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on Mac

# Gated AdaBoost Settings
GATING_K = 3 # Top-k learners to select
GATING_TAU = 1.0 # Temperature for softmax
GATING_LR = 1e-4
GATING_EPOCHS = 20
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
TARGET_TEST_RATIO = 0.9  # 90% of target domain for testing (only 10% for training - limited target data scenario)
