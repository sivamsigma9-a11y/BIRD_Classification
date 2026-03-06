import sys
import os

# Add the project root directory to the Python path
# This ensures we can import from src regardless of where the script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import shutil
from torch.utils.data import DataLoader
from src.config import Config
from src.data import get_dataloaders
from src.model import BasicCNN

def generate_failures():
    print("Generating failure cases...")

    # Define failure directory
    FAILURE_DIR = "failures"
    if os.path.exists(FAILURE_DIR):
        try:
            shutil.rmtree(FAILURE_DIR)
        except OSError as e:
            print(f"Warning: Could not remove directory {FAILURE_DIR}: {e}")
            print("Attempting to continue without removing it...")
            # If we can't remove it, let's try to just use it as is, or maybe clean contents
            # But for now, just proceeding is safer than crashing
    
    os.makedirs(FAILURE_DIR, exist_ok=True)

    # Load resources
    try:
        _, test_loader = get_dataloaders()
    except Exception as e:
        print(f"Error loading dataloaders: {e}")
        return

    dataset = test_loader.dataset
    if hasattr(dataset, 'classes'):
         classes = dataset.classes
    else:
        # Fallback if classes attribute is missing, though ImageFolder has it
        classes = ["Class 0", "Class 1"] 
        print("Warning: classes attribute not found on dataset, using default names.")

    # Check for model existence
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: model.pth not found at {Config.MODEL_PATH}. Please train the model first.")
        return

    # Initialize model
    device = Config.DEVICE
    model = BasicCNN().to(device)
    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()

    failure_count = 0
    total_count = 0

    print(f"Classes found: {classes}")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Assuming binary classification
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float().squeeze()
            
            if predictions.ndim == 0:
                predictions = predictions.unsqueeze(0)

            for i in range(len(labels)):
                true_label_idx = int(labels[i].item())
                pred_label_idx = int(predictions[i].item())
                
                total_count += 1

                if true_label_idx != pred_label_idx:
                    failure_count += 1
                    
                    # Get original file path
                    # Note: indices match dataset.samples ONLY if shuffle=False
                    dataset_idx = batch_idx * test_loader.batch_size + i
                    original_path, _ = dataset.samples[dataset_idx]
                    
                    true_class_name = classes[true_label_idx]
                    pred_class_name = classes[pred_label_idx]
                    
                    fail_dir = os.path.join(FAILURE_DIR, f"{true_class_name}_as_{pred_class_name}")
                    os.makedirs(fail_dir, exist_ok=True)
                    
                    filename = os.path.basename(original_path)
                    shutil.copy2(original_path, os.path.join(fail_dir, filename))
                    
                    print(f"FAILURE: {filename} (True: {true_class_name}, Pred: {pred_class_name})")

    print(f"\nAnalysis complete.")
    print(f"Total images processed: {total_count}")
    print(f"Total failures found: {failure_count}")
    if total_count > 0:
        print(f"Failure rate: {failure_count/total_count:.2%}")
    else:
        print("Failure rate: N/A (0 images)")
    print(f"Failures saved in '{FAILURE_DIR}' directory.")

if __name__ == "__main__":
    generate_failures()
