import torch
import os


class Config:
    # Use absolute paths based on project root to avoid issues when running from different directories
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ORIGINAL_DATA_DIR = os.path.join(PROJECT_ROOT, "New Dataset")
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")

    TRAIN_SPLIT = 0.8
    SEED = 42

    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EXPERIMENT_NAME = "Bird_vs_NoBird_Baseline"
    MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db")
    MLFLOW_ARTIFACT_ROOT = os.path.join(PROJECT_ROOT, "mlruns")
