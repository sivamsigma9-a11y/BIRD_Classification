import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, target_class=0):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # For binary classification with a single output node
        if output.shape[1] == 1:
            score = output
        else:
            score = output[:, target_class]
            
        # Ensure score is a scalar for backward()
        if score.numel() > 1:
            score.sum().backward()
        else:
            score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        return heatmap

    def overlay_heatmap(self, heatmap, original_image_bgr, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_image_bgr.shape[1], original_image_bgr.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)

        # Both heatmap and original_image_bgr are now in BGR
        overlayed_img = cv2.addWeighted(heatmap, alpha, original_image_bgr, 1 - alpha, 0)
        return overlayed_img

    def save_heatmap(self, heatmap_overlay, save_path):
        cv2.imwrite(save_path, heatmap_overlay)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def create_and_save_heatmap(model, target_layer, input_tensor, original_image_path, save_path):
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor)
    
    # Load original image for overlay
    img_bgr = cv2.imread(original_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {original_image_path}")
    
    overlay = grad_cam.overlay_heatmap(heatmap, img_bgr)
    grad_cam.save_heatmap(overlay, save_path)
    grad_cam.remove_hooks()
    return save_path
