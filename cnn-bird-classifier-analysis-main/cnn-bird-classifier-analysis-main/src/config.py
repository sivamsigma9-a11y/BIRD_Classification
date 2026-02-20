import torch


class Config:
    ORIGINAL_DATA_DIR = "original_dataset"
    DATA_DIR = "dataset"

    TRAIN_SPLIT = 0.8
    SEED = 42

    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EXPERIMENT_NAME = "Bird_vs_NoBird_Baseline"
