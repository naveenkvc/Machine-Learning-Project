# train.py
"""
AI-TREAD Deep Transfer Learning Trainer.
Trains pre-trained EfficientNetB0 or MobileNetV2 models with advanced augmentations,
WeightedRandomSampler class balancing, EarlyStopping, ReduceLROnPlateau, 
ModelCheckpoint, and late unfreeze layer-wise fine-tuning.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

class TyreDataset(Dataset):
    """Custom dataset class loading image paths and encoding labels."""
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.classes = ["good", "worn", "damaged"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label_idx = self.data_list[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Create emergency dummy image if file read fails during epoch
            img = Image.new("RGB", (224, 224), color=(0,0,0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label_idx

class EarlyStopping:
    """Closes training loop early if validation loss fails to decrease."""
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"[CALLBACK] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_dataset_splits(data_dir, split_ratio=0.8):
    """Traverses data folders and splits paths into train and val splits."""
    classes = ["good", "worn", "damaged"]
    train_list = []
    val_list = []
    
    for label_idx, c in enumerate(classes):
        folder_path = os.path.join(data_dir, c)
        if not os.path.exists(folder_path):
            continue
            
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle class files
        indices = torch.randperm(len(files)).tolist()
        split_point = int(len(files) * split_ratio)
        
        for i, idx in enumerate(indices):
            item = (files[idx], label_idx)
            if i < split_point:
                train_list.append(item)
            else:
                val_list.append(item)
                
    return train_list, val_list

def main():
    parser = argparse.ArgumentParser(description="AI-TREAD Model Training pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"], help="CNN base skeleton")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for initial classifier training")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Epochs for un-frozen deep fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial Head learning rate")
    parser.add_argument("--fine_tune_lr", type=float, default=5e-5, help="Fine tuning learning rate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE STATUS] Active Training Hardware: '{device}'")
    
    # 1. Split Datasets
    train_list, val_list = get_dataset_splits(args.data_dir)
    if len(train_list) == 0:
        print("[ERROR] No images found. Run 'python dataset_validator.py --generate_synthetic' first.")
        return
        
    print(f"[DATA LIST] Train Samples: {len(train_list)} | Val Samples: {len(val_list)}")
    
    # 2. Setup WeightedRandomSampler for Data Balancing
    labels = [item[1] for item in train_list]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(labels)]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights.numpy(), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # 3. Setup advanced transforms augmentations
    # Apply augmentations aggressively only to train loader
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataloaders
    train_dataset = TyreDataset(train_list, transform=train_transforms)
    val_dataset = TyreDataset(val_list, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. Load Pretrained Core Model
    print(f"[MODEL CORE] Initializing Pretrained '{args.model.upper()}' Backbone...")
    if args.model == "efficientnet":
        # Gracefully handle older torchvision weights syntax
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
        # Adapt linear head classifier for 3 categories with Dropout 0.4
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
    else:
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except Exception:
            model = models.mobilenet_v2(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
        
    model = model.to(device)
    
    # 5. Phase A: Train Head Classifier (Freeze Base Layers)
    print("\n" + "="*50)
    print("PHASE A: INITIAL TRAINING (HEAD ONLY, BASE LAYERS FROZEN)")
    print("="*50)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the classification layer
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Calculate inverse frequency class weights over training set
    train_labels = [item[1] for item in train_list]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = len(train_labels) / (3.0 * class_counts.float())
    class_weights = class_weights.to(device)
    print(f"[CLASS WEIGHTS] Calculated inverse frequency weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    
    # LR Scheduler (Plateau dynamic decay) and EarlyStopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    early_stopping = EarlyStopping(patience=3)
    
    best_loss = float('inf')
    model_save_path = "best_tyre_model.pth"
    
    # Phase A loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total
        
        # Validation epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc:.1f}%) | Val Loss: {epoch_val_loss:.4f} (Acc: {epoch_val_acc:.1f}%)")
        
        # Callbacks
        scheduler.step(epoch_val_loss)
        
        # Model Checkpoint
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"[CHECKPOINT] Saved best classifier model weights to '{model_save_path}'")
            
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("[INFO] Early stopping triggered. Moving to fine-tuning phase.")
            break
            
    # 6. Phase B: Fine-Tuning (Unfreeze final Feature block layers)
    print("\n" + "="*50)
    print("PHASE B: FINE-TUNING (UNFREEZING FINAL FEATURE BLOCKS)")
    print("="*50)
    
    # Reload best checkpoint weights from Phase A
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        
    # Unfreeze the model fully
    for param in model.parameters():
        param.requires_grad = True
        
    # Set a tiny learning rate for deep joint optimization (using AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    early_stopping = EarlyStopping(patience=2)
    
    # Fine-Tuning loop
    for epoch in range(1, args.fine_tune_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        print(f"FT Epoch {epoch}/{args.fine_tune_epochs} | Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc:.1f}%) | Val Loss: {epoch_val_loss:.4f} (Acc: {epoch_val_acc:.1f}%)")
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"[CHECKPOINT] Saved optimized fine-tuned weights to '{model_save_path}'")
            
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("[INFO] Fine-tuning EarlyStopping counter exceeded.")
            break
            
    print(f"\n[SUCCESS] Model Training Pipeline Complete. Optimized Checkpoint: '{model_save_path}'")
    
    # 7. Final Model Evaluation & Metrics Report
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS REPORT (BEST MODEL CHECKPOINT)")
    print("="*50)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    import numpy as np
    confusion_matrix = np.zeros((3, 3), dtype=np.int32)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1
                
    total_samples = np.sum(confusion_matrix)
    correct_samples = np.trace(confusion_matrix)
    overall_acc = 100.0 * correct_samples / total_samples if total_samples > 0 else 0.0
    
    classes_names = ["Good", "Worn", "Damaged"]
    print(f"Overall Dataset Validation Accuracy: {overall_acc:.2f}% ({correct_samples}/{total_samples})")
    print("\nPer-Class Detailed Analysis:")
    
    for i, cname in enumerate(classes_names):
        tp = float(confusion_matrix[i, i])
        fp = float(np.sum(confusion_matrix[:, i]) - tp)
        fn = float(np.sum(confusion_matrix[i, :]) - tp)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"  * Category '{cname.upper()}':")
        print(f"    - Accuracy (Recall): {recall*100:.1f}%")
        print(f"    - Precision:         {precision*100:.1f}%")
        print(f"    - F1-Score:          {f1*100:.1f}%")
        
    print("\nConfusion Matrix:")
    print("              Predicted Good  Predicted Worn  Predicted Damaged")
    print(f"Actual Good     {confusion_matrix[0,0]:<15} {confusion_matrix[0,1]:<15} {confusion_matrix[0,2]:<15}")
    print(f"Actual Worn     {confusion_matrix[1,0]:<15} {confusion_matrix[1,1]:<15} {confusion_matrix[1,2]:<15}")
    print(f"Actual Damaged  {confusion_matrix[2,0]:<15} {confusion_matrix[2,1]:<15} {confusion_matrix[2,2]:<15}")
    print("="*50)

if __name__ == "__main__":
    main()
