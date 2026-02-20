import torch
from .config import Config
from .data import get_dataloaders
from .model import BasicCNN


def evaluate_model():

    _, test_loader = get_dataloaders()

    model = BasicCNN().to(Config.DEVICE)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5

            correct += (predictions.squeeze() == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
