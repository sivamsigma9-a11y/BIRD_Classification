import torch
import mlflow
import mlflow.pytorch
import random
import numpy as np
from torch import nn, optim

import sys
import os

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config
from src.data import get_dataloaders
from src.model import BasicCNN


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model():

    set_seed(Config.SEED)

    train_loader, test_loader = get_dataloaders()

    model = BasicCNN().to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=Config.LEARNING_RATE)

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    with mlflow.start_run():

        mlflow.log_params({
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "image_size": Config.IMAGE_SIZE
        })

        for epoch in range(Config.EPOCHS):

            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(Config.DEVICE)
                labels = labels.float().unsqueeze(1).to(Config.DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            print(f"Epoch [{epoch+1}/{Config.EPOCHS}] "
                  f"Loss: {avg_loss:.4f}")

        mlflow.pytorch.log_model(model, "model")
        
        # Save model locally for easy access by predict.py and evaluate.py
        torch.save(model.state_dict(), "model.pth")


    print("Training completed.")

if __name__ == "__main__":
    train_model()
