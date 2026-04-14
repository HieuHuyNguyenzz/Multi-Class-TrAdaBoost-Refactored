import torch

# General Settings
NUM_FEATURE = 256
NUM_CLASSES = 3
PACKET_NUM = 20
CLIENT_LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 16
SEED = 42

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths (Can be overridden by environment variables)
DATA_DIR = "SOICT Data"
SAME_DIST_PATH = f"{DATA_DIR}/df0.01.feather"
DIFF_1_DIST_PATH = f"{DATA_DIR}/A_data1.feather"
TEST_DIST_PATH = f"{DATA_DIR}/df0.99.feather"
