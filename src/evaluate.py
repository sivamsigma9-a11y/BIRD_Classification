import torch
import mlflow
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


def evaluate_model():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    _, test_loader = get_dataloaders()

    model = BasicCNN().to(Config.DEVICE)
    
    # Check for model existence
    model_path = "model.pth"
    if not os.path.exists(model_path):
        # Try finding it in project root
        potential_path = os.path.join(project_root, "model.pth")
        if os.path.exists(potential_path):
            model_path = potential_path
        else:
            print("Error: model.pth not found in current directory or project root. Please train the model first.")
            return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
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
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    with mlflow.start_run(run_name="Evaluation"):
        mlflow.log_metric("test_accuracy", accuracy)

if __name__ == "__main__":
    evaluate_model()
