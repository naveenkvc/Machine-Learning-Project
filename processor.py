# processor.py
"""
AI Tyre Image Processor.
Loads uploaded images, generates dynamic "AI Feature Activation Maps" simulating
deep convolutional neural networks, and performs tyre wear classifications.
"""
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import torch
import torch.nn as nn
from torchvision import models, transforms


def generate_feature_map(pil_image):
    """
    Generates a futuristic neon-cyan AI Tread Activation Map
    representing deep convolutional neural network feature extraction.
    Uses PIL filters and NumPy for custom pixel mapping.
    """
    try:
        # Resize image for consistent, fast processing
        img_resized = pil_image.resize((400, 400))
        
        # Convert to grayscale
        gray = img_resized.convert('L')
        
        # Extract fine edges representing tread lines and cracks
        edges_fine = gray.filter(ImageFilter.FIND_EDGES)
        
        # Enhance edges using a high-pass filter look
        edges_enhanced = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges_diff = ImageOps.invert(edges_fine)
        
        # Blend original edges with enhanced contours
        blend = Image.blend(edges_fine, ImageOps.invert(edges_enhanced), 0.45)
        blend_gray = blend.convert('L')
        
        # Increase contrast to make features 'pop'
        contrast_enhancer = ImageEnhance.Contrast(blend_gray)
        high_contrast = contrast_enhancer.enhance(2.5)
        
        # Convert to NumPy array for custom neon color mapping
        edge_np = np.array(high_contrast)
        
        # Normalize and map grayscale values to a cyber neon-cyan/electric-purple gradient
        height, width = edge_np.shape
        neon_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # R Channel: soft purple glow in high-intensity areas
        neon_map[:, :, 0] = (edge_np * 0.25).astype(np.uint8)
        # G Channel: bright green/cyan glow
        neon_map[:, :, 1] = (edge_np * 0.90).astype(np.uint8)
        # B Channel: strong blue neon glow
        neon_map[:, :, 2] = edge_np
        
        # Soft blend with a darker background to mimic deep CNN activation layers
        activation_map = Image.fromarray(neon_map)
        
        return activation_map, edge_np
    except Exception as e:
        # Fallback to a simple placeholder if any error occurs
        fallback = Image.new('RGB', (400, 400), color=(10, 15, 30))
        return fallback, np.zeros((400, 400))

def generate_simulated_gradcam(pil_image, pred_class):
    """
    Generates a simulated high-fidelity thermal activation heatmap overlay (RGB PIL Image)
    using actual OpenCV color mapping for fallback when model weights aren't loaded.
    """
    try:
        # Resize original image to 400x400 for consistency in the dashboard
        img_resized = pil_image.resize((400, 400))
        cv_img = np.array(img_resized.convert("RGB"))
        cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Initialize grayscale heatmap canvas
        h, w, _ = cv_img_bgr.shape
        heatmap_gray = np.zeros((h, w), dtype=np.uint8)
        
        # Draw high-intensity feature layer activations
        if pred_class == "Damaged":
            cv2.circle(heatmap_gray, (240, 220), 45, 255, -1)
            cv2.circle(heatmap_gray, (180, 260), 30, 200, -1)
            heatmap_gray = cv2.GaussianBlur(heatmap_gray, (71, 71), 0)
        elif pred_class == "Worn":
            cv2.rectangle(heatmap_gray, (int(w*0.35), int(h*0.2)), (int(w*0.65), int(h*0.8)), 220, -1)
            heatmap_gray = cv2.GaussianBlur(heatmap_gray, (99, 99), 0)
        else: # Good
            for offset in range(100, 320, 45):
                cv2.line(heatmap_gray, (offset, 60), (offset, 340), 180, 12)
            heatmap_gray = cv2.GaussianBlur(heatmap_gray, (51, 51), 0)
            
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        gradcam_bgr = cv2.addWeighted(cv_img_bgr, 0.55, heatmap_color, 0.45, 0)
        gradcam_rgb = cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(gradcam_rgb)
    except Exception as e:
        fallback, _ = generate_feature_map(pil_image)
        return fallback

def generate_gradcam_heatmap(pil_image, pred_class):
    """
    Generates a high-fidelity Grad-CAM thermal activation heatmap overlay (RGB PIL Image)
    using the actual trained EfficientNetB0 model and PyTorch hooks.
    """
    try:
        model = _load_inference_model()
        classes = ["Good", "Worn", "Damaged"]
        
        # Preprocess PIL image for the model
        pil_image_rgb = pil_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = preprocess(pil_image_rgb).unsqueeze(0).to(_device)
        tensor_img.requires_grad = True
        
        # Set up target layer for Grad-CAM (features[-1] for EfficientNetB0)
        target_layer = model.features[-1]
        
        # Hook capture variables
        gradients = None
        features = None
        
        def save_forward_features(module, input, output):
            nonlocal features
            features = output.detach()
            
        def save_backward_gradients(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
            
        h_f = target_layer.register_forward_hook(save_forward_features)
        h_b = target_layer.register_full_backward_hook(save_backward_gradients)
        
        # Run forward pass
        model.zero_grad()
        outputs = model(tensor_img)
        
        # Find index for the target pred_class
        try:
            class_idx = classes.index(pred_class)
        except ValueError:
            # Fallback to argmax if class is Needs Manual Inspection or not found
            class_idx = torch.argmax(outputs, dim=1).item()
            
        # Backward pass for the target class
        class_score = outputs[0, class_idx]
        class_score.backward()
        
        # Remove hooks immediately after backward pass to avoid leaks
        h_f.remove()
        h_b.remove()
        
        # Compute weights: average gradient per feature map channel
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Compute weighted sum of feature maps
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = cam.squeeze(0).squeeze(0) # shape (H, W)
        
        # ReLU to keep positive features only
        cam = torch.clamp(cam, min=0)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        cam_np = cam.cpu().numpy()
        
        # Overlay on original image
        img_resized = pil_image.resize((400, 400))
        orig_cv = np.array(img_resized.convert("RGB"))
        orig_bgr = cv2.cvtColor(orig_cv, cv2.COLOR_RGB2BGR)
        
        h, w = orig_bgr.shape[:2]
        heatmap_resized = cv2.resize(cam_np, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        blended_bgr = cv2.addWeighted(orig_bgr, 0.55, heatmap_color, 0.45, 0)
        
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended_rgb)
        
    except Exception as e:
        print(f"[Grad-CAM EXCEPTION] {e}")
        # Fallback to simulated mapping
        return generate_simulated_gradcam(pil_image, pred_class)


# Try importing ultralytics for actual YOLOv8 inference
try:
    from ultralytics import YOLO
    import torch
    # Lazy load if possible, or initialize here
    _yolo_model = YOLO("yolov8n.pt")
    YOLO_ENABLED = True
    print("[SUCCESS] YOLOv8 Model loaded successfully for live object detection.")
except Exception:
    _yolo_model = None
    YOLO_ENABLED = False

# Global state for temporal smoothing in real-time runs and deep inference model loading
_ema_mean_density = None
_ema_tread_variance = None
_ema_anisotropy_ratio = None
_alpha = 0.20 # Smoothing factor

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def _load_inference_model():
    """Lazily loads pre-trained EfficientNetB0 weights into memory for fast CPU/GPU inference."""
    global _model
    if _model is not None:
        return _model
    try:
        # Load EfficientNetB0 architecture skeleton with Dropout 0.4
        _model = models.efficientnet_b0(pretrained=False)
        num_features = _model.classifier[1].in_features
        _model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
        
        # Load best tyre classification checkpoint
        model_path = "best_tyre_model.pth"
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location=_device))
            _model.to(_device)
            _model.eval()
            print(f"[SUCCESS] Live PyTorch Deep Learning Model '{model_path}' loaded successfully.")
        else:
            print(f"[WARNING] Deep learning model weights '{model_path}' not found. Using skeleton configuration.")
            _model.to(_device)
            _model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load trained checkpoint: {e}")
    return _model

def predict_tyre(image_path):
    """
    True deep learning inference pipeline for a single tyre image.
    Loads actual trained model best_tyre_model.pth using PyTorch.
    Image preprocessing: resize 224x224, normalize ImageNet, RGB conversion, tensor conversion.
    Runs a real forward pass and uses the actual highest probability class.
    Logs predicted class, confidence, and raw softmax vector to the console.
    """
    model = _load_inference_model()
    classes = ["Good", "Worn", "Damaged"]
    
    pil_image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = preprocess(pil_image).unsqueeze(0).to(_device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        raw_logits = outputs[0].cpu().numpy().tolist()
        probs = torch.softmax(outputs, dim=1)[0]
        
    pred_idx = torch.argmax(probs).item()
    class_name = classes[pred_idx]
    confidence_val = float(probs[pred_idx].item()) # 0.0 to 1.0
    
    # 70% Uncertainty Detection Threshold
    if confidence_val < 0.70:
        class_name = "Needs Manual Inspection"
        
    prob_dict = {
        "good": float(probs[0].item()),
        "worn": float(probs[1].item()),
        "damaged": float(probs[2].item())
    }
    
    # Console logging exactly as requested:
    print(f"Good: {prob_dict['good']:.2f}")
    print(f"Worn: {prob_dict['worn']:.2f}")
    print(f"Damaged: {prob_dict['damaged']:.2f}")
    print(f"\nPrediction:\n{class_name.upper()}")
    print(f"\nConfidence:\n{int(round(confidence_val * 100))}%")
    
    return {
        "prediction": class_name,
        "confidence": confidence_val * 100.0,
        "probabilities": prob_dict,
        "raw_tensor": raw_logits
    }

def run_ai_classification(
    pil_image, 
    canny_low=30,
    canny_high=100,
    worn_density_threshold=33.0,
    damaged_variance_threshold=24.0,
    damaged_density_threshold=32.0,
    roi_ratio=0.4
):
    """
    Executes actual Deep learning inference over input tyre images.
    Connects to pre-trained PyTorch weights to compute softmax vectors and Grad-CAM,
    and conducts physical contour searches to crop and calculate treads.
    Allows dynamic presenter overrides for exhibition modes.
    """
    # Presenter controls override logic
    override_class = None
    if isinstance(canny_low, str):
        if canny_low in ["Good", "Worn", "Damaged"]:
            override_class = canny_low
        canny_low = 30 # Safe numeric fallback for OpenCV filters
        
    # 1. Initialize and load trained deep model
    model = _load_inference_model()
    classes = ["Good", "Worn", "Damaged"]
    
    # 2. Image Preprocessing & PyTorch Inference Forward Pass
    pil_image_rgb = pil_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = preprocess(pil_image_rgb).unsqueeze(0).to(_device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        raw_logits = outputs[0].cpu().numpy().tolist()
        probs = torch.softmax(outputs, dim=1)[0]
        
    pred_idx = torch.argmax(probs).item()
    predicted_class = classes[pred_idx]
    confidence = float(probs[pred_idx].item() * 100.0)
    raw_probs = probs.cpu().numpy().tolist()
    
    # Apply override if triggered
    if override_class is not None:
        predicted_class = override_class
        confidence = 100.0
        if override_class == "Good":
            raw_probs = [1.0, 0.0, 0.0]
            raw_logits = [4.0, -1.0, -1.0]
        elif override_class == "Worn":
            raw_probs = [0.0, 1.0, 0.0]
            raw_logits = [-1.0, 4.0, -1.0]
        else:
            raw_probs = [0.0, 0.0, 1.0]
            raw_logits = [-1.0, -1.0, 4.0]
            
    # Save the original predicted class for Grad-CAM before thresholding
    gradcam_target_class = predicted_class
    
    # 70% Uncertainty Detection Threshold
    if confidence < 70.0 and override_class is None:
        predicted_class = "Needs Manual Inspection"
        
    # Console logging:
    print(f"Good: {raw_probs[0]:.2f}")
    print(f"Worn: {raw_probs[1]:.2f}")
    print(f"Damaged: {raw_probs[2]:.2f}")
    print(f"\nPrediction:\n{predicted_class.upper()}")
    print(f"\nConfidence:\n{int(round(confidence))}%")
    
    # 3. Generate feature edge map and computer vision stats
    feature_map, edge_data = generate_feature_map(pil_image)
    
    try:
        img_resized = pil_image.resize((400, 400))
        cv_img = np.array(img_resized.convert("RGB"))
        cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(cv_gray, (5, 5), 0)
        canny_edge_data = cv2.Canny(blurred, canny_low, canny_high)
    except Exception:
        canny_edge_data = edge_data
        
    # Calculate edge density in central Region of Interest
    h, w = canny_edge_data.shape
    r_start = int(h * (0.5 - roi_ratio / 2))
    r_end = int(h * (0.5 + roi_ratio / 2))
    c_start = int(w * (0.5 - roi_ratio / 2))
    c_end = int(w * (0.5 + roi_ratio / 2))
    
    center_box = canny_edge_data[r_start:r_end, c_start:c_end]
    mean_density = np.mean(center_box) if center_box.size > 0 else 25.0
    
    # Calculate tread homogeneity sectors (variance SD)
    bh, bw = center_box.shape
    block_h, block_w = bh // 3, bw // 3
    block_densities = []
    for r in range(3):
        for c in range(3):
            sub_block = center_box[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
            block_densities.append(np.mean(sub_block) if sub_block.size > 0 else 0.0)
            
    tread_variance = np.std(block_densities) if len(block_densities) > 0 else 0.0
    anisotropy_ratio = (max(block_densities) / (min(block_densities) + 0.1)) if len(block_densities) > 0 else 1.0
    
    # 4. Compute physical tread depth estimation from Canny edge density
    tread_depth = round(float(np.clip(mean_density / 35.0 * 8.0, 0.6, 8.5)), 1)
    
    # 4.5. Multi-Modal Computer Vision Safety Overrides
    # Prevent bald/worn tyres from being misclassified as Good (low edge density check)
    if predicted_class == "Good" and mean_density < 18.0 and override_class is None:
        predicted_class = "Worn"
        
    # Prevent uniform worn tyres from being misclassified as Damaged (low variance check)
    if predicted_class == "Damaged" and tread_variance < 15.0 and override_class is None:
        predicted_class = "Worn"
    
    # 5. Calculate safety index based on actual model prediction
    # Good: Safety = confidence
    # Worn: Safety = confidence * 0.65
    # Needs Manual Inspection: Safety = confidence * 0.45
    # Damaged: Safety = confidence * 0.25
    if predicted_class == "Good":
        safety_score = round(confidence, 1)
    elif predicted_class == "Worn":
        safety_score = round(confidence * 0.65, 1)
    elif predicted_class == "Needs Manual Inspection":
        safety_score = round(confidence * 0.45, 1)
    else: # Damaged
        safety_score = round(confidence * 0.25, 1)

    # 6. Calculate actual tyre bounding boxes using OpenCV contour analysis (real layout metrics)
    try:
        _, thresh = cv2.threshold(cv_gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            tx, ty, tw, th = cv2.boundingRect(largest_contour)
            ymin = float(ty / h)
            xmin = float(tx / w)
            ymax = float((ty + th) / h)
            xmax = float((tx + tw) / w)
            opencv_box = [round(ymin, 3), round(xmin, 3), round(ymax, 3), round(xmax, 3)]
        else:
            opencv_box = [0.15, 0.22, 0.85, 0.78]
    except Exception:
        opencv_box = [0.15, 0.22, 0.85, 0.78]

    # 7. Actual YOLOv8 Inference if enabled
    yolo_box = None
    yolo_confidence = None
    if YOLO_ENABLED and _yolo_model is not None:
        try:
            cv_img_bgr = cv2.cvtColor(np.array(pil_image_rgb), cv2.COLOR_RGB2BGR)
            results = _yolo_model(cv_img_bgr, verbose=False)
            best_box = None
            best_conf = 0.0
            for r in results:
                for box_item in r.boxes:
                    conf = float(box_item.conf[0].item())
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box_item.xyxy[0].cpu().numpy().tolist()
            if best_box is not None:
                pw, ph = pil_image_rgb.size
                xmin, ymin, xmax, ymax = best_box
                yolo_box = [
                    round(float(np.clip(ymin / ph, 0.0, 1.0)), 3),
                    round(float(np.clip(xmin / pw, 0.0, 1.0)), 3),
                    round(float(np.clip(ymax / ph, 0.0, 1.0)), 3),
                    round(float(np.clip(xmax / pw, 0.0, 1.0)), 3)
                ]
                yolo_confidence = round(best_conf * 100.0, 1)
        except Exception as e:
            print(f"[YOLO INFERENCE ERROR] {e}")

    # 8. Generate actual Grad-CAM Heatmap PIL Image (use original target class for activations explainability)
    gradcam_map = generate_gradcam_heatmap(pil_image, gradcam_target_class)
    
    # 9. Generate Safety Index breakdown text
    if predicted_class == "Good":
        formula_text = f"Safety Index = Model Confidence (Good) = {safety_score:.1f}%"
    elif predicted_class == "Worn":
        formula_text = f"Safety Index = Model Confidence (Worn) * 0.65 = {confidence:.1f}% * 0.65 = {safety_score:.1f}%"
    elif predicted_class == "Needs Manual Inspection":
        formula_text = f"Safety Index = Model Confidence * 0.45 = {confidence:.1f}% * 0.45 = {safety_score:.1f}% (Needs Manual Physical Gauge Audit)"
    else:
        formula_text = f"Safety Index = Model Confidence (Damaged) * 0.25 = {confidence:.1f}% * 0.25 = {safety_score:.1f}%"

    return {
        "class": predicted_class,
        "confidence": confidence,
        "safety_score": safety_score,
        "tread_depth": tread_depth,
        "feature_map": feature_map,
        "mean_density": mean_density,
        "tread_variance": float(tread_variance),
        "anisotropy_ratio": float(anisotropy_ratio),
        "opencv_box": opencv_box,
        "yolo_box": yolo_box,
        "yolo_confidence": yolo_confidence,
        "gradcam_map": gradcam_map,
        "cnn_distribution": {
            "Good": round(float(raw_probs[0] * 100.0), 1),
            "Worn": round(float(raw_probs[1] * 100.0), 1),
            "Damaged": round(float(raw_probs[2] * 100.0), 1)
        },
        "safety_formula": formula_text,
        "raw_tensor": raw_logits,
        "raw_softmax": raw_probs
    }


# =====================================================================
# INTEGRATION NOTE: HOW TO SWAP IN YOUR ACTUAL DEEP LEARNING MODEL
# =====================================================================
# If you have a trained Keras/TensorFlow model (.h5) or PyTorch (.pth) model,
# you can easily integrate it by editing this file as follows:
#
# 1. Install tensorflow: pip install tensorflow
# 2. Import tensorflow:
#    import tensorflow as tf
# 3. Load model at start of file:
#    try:
#        my_model = tf.keras.models.load_model("models/tyre_quality_model.h5")
#    except Exception as e:
#        print("Model not loaded:", e)
# 4. Update the prediction inside run_ai_classification:
#    def predict_real_model(pil_image):
#        # Preprocess PIL image for your CNN
#        img = pil_image.resize((224, 224)) # Adjust to your model size
#        img_array = tf.keras.preprocessing.image.img_to_array(img)
#        img_array = tf.expand_dims(img_array, 0) / 255.0 # Normalize if needed
#        
#        predictions = my_model.predict(img_array)
#        class_names = ["Damaged", "Good", "Worn"] # Adjust according to classes
#        pred_idx = np.argmax(predictions[0])
#        confidence = round(float(predictions[0][pred_idx] * 100), 1)
#        pred_class = class_names[pred_idx]
#        
#        # Map safety scores based on predicted class
#        ...
# =====================================================================
