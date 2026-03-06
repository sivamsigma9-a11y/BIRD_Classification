import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageEnhance

# Allow standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import BasicCNN
from src.config import Config

# ---------------- SETTINGS ----------------
SAVE_DIR = "filter_maps"

IMAGE_PATH = None
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    print("Usage: python stress_test.py <image_path>")
    sys.exit(1)
# ---------------- PATH CHECKS ----------------
if not os.path.exists(Config.MODEL_PATH):
    print(f"Error: model.pth not found at {Config.MODEL_PATH}. Train model first.")
    sys.exit(1)

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found → {IMAGE_PATH}")
    sys.exit(1)

os.makedirs(SAVE_DIR, exist_ok=True)

device = Config.DEVICE

# ---------------- LOAD MODEL ----------------
print(f"Loading model from {Config.MODEL_PATH}...")
model = BasicCNN().to(device)
try:
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

# ---------------- CORRUPTION ----------------
def add_noise(img, intensity):
    arr = np.array(img).astype(np.int16)

    # symmetric noise (more realistic)
    noise = np.random.randint(-intensity, intensity + 1, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

def add_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

# ---------------- FEATURE MAP HOOKS ----------------
feature_maps = {}

def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook

# Register hooks (assuming Sequential layout)
model.features[0].register_forward_hook(get_activation("conv1"))
model.features[3].register_forward_hook(get_activation("conv2"))
model.features[6].register_forward_hook(get_activation("conv3"))

# ---------------- VISUALIZATION ----------------
def plot_feature_maps(fmap, step, layer_name):
    fmap = fmap.squeeze(0)
    num_filters = fmap.shape[0]

    cols = int(np.ceil(np.sqrt(num_filters)))
    rows = int(np.ceil(num_filters / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    plt.suptitle(f"{layer_name} - Step {step}", fontsize=14)

    for i in range(num_filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap[i].cpu(), cmap='viridis')
        plt.axis('off')
        plt.title(f"{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/step{step}_{layer_name}.png")
    plt.close()

def show_prediction(image, prediction, confidence, step):
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis('off')
    plt.title(
        f"Step {step}\nPrediction: {prediction}\nConfidence: {confidence:.4f}",
        fontsize=10
    )
    # plt.show()

def save_corrupted_image(image, step):
    step_dir = os.path.join(SAVE_DIR, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image - Step {step}")

    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, "corrupted.png"))
    plt.close()

# ---------------- LOAD IMAGE ----------------
original_img = Image.open(IMAGE_PATH).convert("RGB")

print("\n=== Progressive Failure Stress Test ===")

for step in range(1, 6):
    corrupted = original_img.copy()

    if step == 1:
        noise_level = 0
        brightness_bias = 1.0
        print(f"\nStep {step} (Clean Image)")
    else:
        noise_level = step * 20
        brightness_bias = 1 + step * 0.3

        corrupted = add_noise(corrupted, noise_level)
        corrupted = add_brightness(corrupted, brightness_bias)

        print(f"\nStep {step}")
        print(f"Noise Level: {noise_level}")
        print(f"Brightness Bias: {brightness_bias:.2f}")

    input_tensor = transform(corrupted).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Bird" if prob > 0.5 else "No Bird"

    print(f"Prediction: {prediction}")
    print(f"Confidence: {prob:.4f}")

    save_corrupted_image(corrupted, step)

    plot_feature_maps(feature_maps["conv1"], step, "Conv1 (16 filters)")
    plot_feature_maps(feature_maps["conv2"], step, "Conv2 (32 filters)")
    plot_feature_maps(feature_maps["conv3"], step, "Conv3 (64 filters)")

print(f"\n[SUCCESS] All outputs saved in: {os.path.abspath(SAVE_DIR)}")
