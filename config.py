import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/test"
LEARNING_RATE = 0.00001
BATCH_SIZE = 8
NUM_WORKERS = 8
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pt"
CHECKPOINT_GEN = "gen.pt"
REFINE_CHECKPOINT_DISC = "ref_disc.pt"
REFINE_CHECKPOINT_GEN = "ref_gen.pt"
MS_SSIM_LAMBDA = 10
