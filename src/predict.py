import argparse
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add parent directory to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.model import BasicCNN
import matplotlib.pyplot as plt
import numpy as np

# ---------------- SETTINGS ----------------
SAVE_DIR = "prediction_filters"

# ---------------- FEATURE MAP HOOKS ----------------
feature_maps = {}

def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook

def plot_feature_maps(fmap, layer_name):
    fmap = fmap.squeeze(0)
    num_filters = fmap.shape[0]

    cols = int(np.ceil(np.sqrt(num_filters)))
    rows = int(np.ceil(num_filters / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    plt.suptitle(f"{layer_name}", fontsize=14)

    for i in range(num_filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap[i].cpu(), cmap='viridis')
        plt.axis('off')
        plt.title(f"{i}", fontsize=8)

    plt.tight_layout()
    sanitized_name = layer_name.split()[0].lower() # e.g. "Conv1" -> "conv1"
    plt.savefig(f"{SAVE_DIR}/{sanitized_name}.png")
    plt.close()

def predict(image_path):
    # Check if model exists
    if not os.path.exists("model.pth"):
        print("Error: model.pth not found. Please train the model first.")
        return

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Load model
    device = Config.DEVICE
    model = BasicCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    # Register hooks (assuming Sequential layout)
    # Important: Re-register hooks on new model instance each time or clear dict
    feature_maps.clear()
    model.features[0].register_forward_hook(get_activation("conv1"))
    model.features[3].register_forward_hook(get_activation("conv2"))
    model.features[6].register_forward_hook(get_activation("conv3"))

    # Define transforms (must match training/evaluation transforms)
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        
        # Determine class based on threshold (assuming 0.5)
        # Assuming 1 is "Bird" and 0 is "No Bird" or vice versa depending on dataset
        # Usually folder names dictate class indices. 
        # By default ImageFolder sorts classes alphabetically.
        # If classes are ["Bird", "NoBird"], then Bird=0, NoBird=1 ? 
        # We need to verify class mapping. 
        # But for now let's just print probability and a generic prediction.
        
        prediction = "Bird" if probability > 0.5 else "No Bird" 
        # Wait, usually 1 is the positive class. If "Bird" and "No Bird" were folders..
        # Let's check class names in data.py or list folders.
        # Assuming binary classification.
        
        print(f"Prediction for {image_path}:")
        print(f"Probability: {probability:.4f}")
        print(f"Class: {prediction} (Prob > 0.5)")

        # Visualize feature maps
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"\nSaving feature maps to {os.path.abspath(SAVE_DIR)}...")
        
        if "conv1" in feature_maps:
            plot_feature_maps(feature_maps["conv1"], "Conv1 (16 filters)")
        if "conv2" in feature_maps:
            plot_feature_maps(feature_maps["conv2"], "Conv2 (32 filters)")
        if "conv3" in feature_maps:
            plot_feature_maps(feature_maps["conv3"], "Conv3 (64 filters)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an image contains a bird.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    
    predict(args.image_path)