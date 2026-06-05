# gradcam.py
"""
AI-TREAD Grad-CAM Explainer.
Extracts deep gradient feature activation mappings from final convolutional layers 
of pre-trained MobileNetV2 or EfficientNetB0 models and exports visual overlays.
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class PyTorchGradCAM:
    """
    Coordinates forward/backward activation hooks to extract class-specific 
    gradient heatmaps from PyTorch convolutional layers.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_forward_features)
        self.target_layer.register_full_backward_hook(self._save_backward_gradients)

    def _save_forward_features(self, module, input, output):
        self.features = output.detach()

    def _save_backward_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """Generates a normalized grayscale heatmap for a specific target class."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Compute weighted combination of forward features
        cam = torch.sum(weights * self.features, dim=1, keepdim=True)
        cam = cam.squeeze(0).squeeze(0) # shape (H, W)
        
        # Apply ReLU to retain only positive activation features
        cam = torch.clamp(cam, min=0)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.cpu().numpy(), class_idx

def overlay_heatmap(original_img_path, heatmap_gray, output_path=None, alpha=0.45):
    """Blends BGR JET heatmap over the raw original input image."""
    try:
        orig = cv2.imread(original_img_path)
        if orig is None:
            raise FileNotFoundError(f"Original image not found: {original_img_path}")
            
        h, w = orig.shape[:2]
        
        # Resize grayscale heatmap to match raw image dimensions
        heatmap_resized = cv2.resize(heatmap_gray, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply JET colormap for hot-to-cold gradient mapping
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Blend overlay
        blended = cv2.addWeighted(orig, 1.0 - alpha, heatmap_color, alpha, 0)
        
        if output_path:
            cv2.imwrite(output_path, blended)
            print(f"[SUCCESS] Grad-CAM saved to: {output_path}")
            
        return blended
    except Exception as e:
        print(f"[ERROR] Overlay failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualizer tool.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input tyre image")
    parser.add_argument("--model_path", type=str, required=True, help="Trained .pth model checkpoint path")
    parser.add_argument("--output_path", type=str, default="gradcam_output.png", help="Output path for blended image")
    parser.add_argument("--model_type", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"], help="Model architecture")
    args = parser.parse_args()
    
    # 1. Initialize empty PyTorch model skeleton to load weights into
    if args.model_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
        target_layer = model.features[-1]
    else:
        model = models.mobilenet_v2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
        target_layer = model.features[-1]
        
    # Load model checkpoint
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        print(f"[SUCCESS] Loaded trained weights from '{args.model_path}'")
    else:
        print(f"[WARNING] Weights path not found. Running with uninitialized random parameters.")
        
    # 2. Image Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        pil_img = Image.open(args.image_path).convert("RGB")
        tensor_img = preprocess(pil_img).unsqueeze(0)
    except Exception as e:
        print(f"[ERROR] Image loading error: {e}")
        return
        
    # 3. Compute Grad-CAM
    cam_explainer = PyTorchGradCAM(model, target_layer)
    heatmap_gray, class_idx = cam_explainer.generate_heatmap(tensor_img)
    
    classes = ["Good", "Worn", "Damaged"]
    print(f"[Grad-CAM] Maximum activations calculated for Target Label: '{classes[class_idx]}'")
    
    # 4. Save Overlay
    overlay_heatmap(args.image_path, heatmap_gray, args.output_path)

if __name__ == "__main__":
    main()
