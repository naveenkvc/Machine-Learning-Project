# confusion_matrix.py
"""
AI-TREAD Confusion Matrix Generator.
Constructs a visual 3x3 grid matrix of model classification accuracies 
and saves the publication-grade glowing heatmap to disk as 'confusion_matrix.png'.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend for remote containers
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
from train import TyreDataset, get_dataset_splits

def plot_confusion_matrix(matrix, classes, output_path="confusion_matrix.png"):
    """
    Renders a stunning glowing neon-style confusion matrix heatmap 
    with values printed inside each cell.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Futuristic dark facecolors
    fig.patch.set_facecolor('#0b0e14')
    ax.set_facecolor('#0b0e14')
    
    # Display the pixel matrix using an electric color map
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Purples)
    
    # Custom colored colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Axis ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold', color='#00f2fe')
    ax.set_yticklabels(classes, fontsize=10, fontweight='bold', color='#00f2fe')
    
    # Rotate x labels for clean visuals
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Print quantitative value inside each cell
    threshold = matrix.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = "white" if matrix[i, j] < threshold else "black"
            ax.text(j, i, f"{matrix[i, j]}",
                    ha="center", va="center",
                    color=text_color, fontsize=12, fontweight='bold')
            
    # Set titles with glowing cyan colors
    ax.set_title("AI-TREAD // Confusion Matrix", fontsize=14, fontweight='black', pad=15, color='#ffffff')
    ax.set_xlabel("Predicted Categories", fontsize=11, fontweight='bold', labelpad=12, color='#94a3b8')
    ax.set_ylabel("True Category Labels", fontsize=11, fontweight='bold', labelpad=12, color='#94a3b8')
    
    # Set border line colors
    ax.spines['top'].set_color('#1e293b')
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.spines['right'].set_color('#1e293b')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Glowing Confusion Matrix saved as: '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Generate and plot Model Confusion Matrix")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--model_path", type=str, default="best_tyre_model.pth", help="Checkpoint weights path")
    parser.add_argument("--model_type", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"], help="CNN base type")
    parser.add_argument("--output_path", type=str, default="confusion_matrix.png", help="Output PNG path")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize core architecture structure
    if args.model_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    else:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[SUCCESS] Model loaded from checkpoint: '{args.model_path}'")
    else:
        print(f"[ERROR] Checkpoint weights not found at '{args.model_path}'")
        return
        
    model = model.to(device)
    model.eval()
    
    # 2. Get splits
    _, val_list = get_dataset_splits(args.data_dir)
    if len(val_list) == 0:
        print("[ERROR] Verification subset splits empty.")
        return
        
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = TyreDataset(val_list, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    classes = ["Good", "Worn", "Damaged"]
    matrix = np.zeros((3, 3), dtype=np.int32)
    
    # 3. Gather predictions
    with torch.no_grad():
        for input_tensor, target in val_loader:
            input_tensor = input_tensor.to(device)
            target_idx = target.item()
            
            outputs = model(input_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            matrix[target_idx, pred_idx] += 1
            
    # 4. Generate visual plot matrix
    plot_confusion_matrix(matrix, classes, args.output_path)

if __name__ == "__main__":
    main()
