"""Configuration for Text2Gloss training"""
import torch

# Model configuration
MODEL_CHECKPOINT = "facebook/mbart-large-50"
SOURCE_LANG = "vi_VN"
TARGET_LANG = "vi_VN"

# Data configuration
DATA_FILE = "./text2gloss/data/Corpus-Vie-VSL-10K.xlsx"
MAX_LENGTH = 128
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1

# Training hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.2
DROPOUT_RATE = 0.3

# System configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True if DEVICE == "cuda" else False

# Model output
MODEL_OUTPUT_DIR = "./text2gloss/models/text2gloss_model"
SAVE_TOTAL_LIMIT = 3
