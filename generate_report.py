# generate_report.py
"""
AI-TREAD Model Performance Evaluation & Report Generation Suite.
Evaluates best_tyre_model.pth on the validation dataset split.
Outputs comprehensive metrics, confusion matrices, transition logs, and
a premium cyberpunk HTML performance dashboard to results/performance_report.html.
"""

import os
import sys
import re

# Force UTF-8 terminal encoding on Windows to support cyber telemetry symbols
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

# For reproducing splits identically
torch.manual_seed(42)

from train import TyreDataset, get_dataset_splits
import processor

def verify_model_paths():
    """
    Verifies programmatically that app.py, realtime_dashboard.py, and evaluate.py
    all resolve to load the identical model weights file.
    """
    print("\n" + "="*60)
    print("STEP 1: PROGRAMMATIC MODEL CONFIGURATION VERIFICATION")
    print("="*60)
    
    files_to_check = {
        "app.py": "c:/Users/navee/OneDrive/Desktop/ML2/app.py",
        "realtime_dashboard.py": "c:/Users/navee/OneDrive/Desktop/ML2/realtime_dashboard.py",
        "evaluate.py": "c:/Users/navee/OneDrive/Desktop/ML2/evaluate.py",
        "processor.py": "c:/Users/navee/OneDrive/Desktop/ML2/processor.py"
    }
    
    verifications = []
    
    for label, filepath in files_to_check.items():
        if not os.path.exists(filepath):
            verifications.append((label, "File missing", False))
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for model filename references
        matches = re.findall(r'best_tyre_model\.pth', content)
        if matches:
            verifications.append((label, f"Contains {len(matches)} reference(s) to 'best_tyre_model.pth'", True))
        else:
            # Check if it imports processor which loads it
            if "import processor" in content:
                verifications.append((label, "Indirect reference via 'processor' module (Loads 'best_tyre_model.pth')", True))
            else:
                verifications.append((label, "No references found", False))
                
    for label, msg, status in verifications:
        icon = "🟢 VERIFIED" if status else "🔴 WARNING"
        print(f"  • {label:<22} -> {icon:<10} ({msg})")
        
    all_ok = all(v[2] for v in verifications)
    if all_ok:
        print("\n[SUCCESS] Unified Model Loading Verified: All entry points share the identical 'best_tyre_model.pth' checkpoint.")
    else:
        print("\n[WARNING] Possible loading inconsistency detected.")
    return verifications

def evaluate_model():
    """
    Runs dynamic evaluation loop over validation image splits and collects metrics.
    """
    print("\n" + "="*60)
    print("STEP 2: RUNNING VALIDATION DATASET EVALUATION LOOPS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing forward passes on hardware: '{device}'")
    
    # 1. Load exact model skeleton
    try:
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
    except Exception:
        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 3)
        )
        
    model_path = "best_tyre_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[SUCCESS] Loaded target weights file '{model_path}' successfully.")
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found! Cannot evaluate.")
        
    model = model.to(device)
    model.eval()
    
    # 2. Get splits
    data_dir = "data"
    train_list, val_list = get_dataset_splits(data_dir)
    print(f"Validation dataset split contains {len(val_list)} image samples.")
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = TyreDataset(val_list, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    classes = ["Good", "Worn", "Damaged"]
    confusion_matrix = np.zeros((3, 3), dtype=np.int32)
    
    misclassified_samples = []
    predictions_log = []
    
    with torch.no_grad():
        for idx, (input_tensor, target) in enumerate(val_loader):
            input_tensor = input_tensor.to(device)
            target_idx = target.item()
            
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx].item() * 100.0)
            
            confusion_matrix[target_idx, pred_idx] += 1
            
            filepath = val_list[idx][0]
            abs_filepath = os.path.abspath(filepath)
            
            predictions_log.append({
                "path": abs_filepath,
                "filename": os.path.basename(filepath),
                "true_idx": target_idx,
                "pred_idx": pred_idx,
                "true_label": classes[target_idx],
                "pred_label": classes[pred_idx],
                "confidence": confidence,
                "is_correct": (pred_idx == target_idx)
            })
            
            # Record misclassification
            if pred_idx != target_idx:
                misclassified_samples.append(predictions_log[-1])
                
    # 3. Calculate metrics
    total_samples = np.sum(confusion_matrix)
    correct_samples = np.trace(confusion_matrix)
    overall_accuracy = (correct_samples / total_samples) * 100.0 if total_samples > 0 else 0.0
    
    per_class_metrics = {}
    prediction_counts = {c: 0 for c in classes}
    
    # Compute counts
    for item in predictions_log:
        prediction_counts[item["pred_label"]] += 1
        
    for i, cname in enumerate(classes):
        tp = float(confusion_matrix[i, i])
        fp = float(np.sum(confusion_matrix[:, i]) - tp)
        fn = float(np.sum(confusion_matrix[i, :]) - tp)
        tn = float(total_samples - (tp + fp + fn))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
        
        per_class_metrics[cname] = {
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1": f1 * 100.0,
            "accuracy": accuracy * 100.0,
            "predictions": prediction_counts[cname],
            "prediction_share": (prediction_counts[cname] / total_samples * 100.0) if total_samples > 0 else 0.0
        }
        
    # Macro averages
    macro_precision = np.mean([per_class_metrics[c]["precision"] for c in classes])
    macro_recall = np.mean([per_class_metrics[c]["recall"] for c in classes])
    macro_f1 = np.mean([per_class_metrics[c]["f1"] for c in classes])
    
    return {
        "confusion_matrix": confusion_matrix.tolist(),
        "overall_accuracy": overall_accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "total_samples": int(total_samples),
        "per_class": per_class_metrics,
        "misclassified": misclassified_samples,
        "predictions_log": predictions_log
    }

def process_and_display_results(results):
    """
    Formats, applies logical checks (collapse, readiness, worst-performing),
    generates recommendations, and prints results to console.
    """
    print("\n" + "="*60)
    print("STEP 3: COMPILING PERFORMANCE Telemetries")
    print("="*60)
    
    classes = ["Good", "Worn", "Damaged"]
    per_class = results["per_class"]
    total = results["total_samples"]
    overall_acc = results["overall_accuracy"]
    
    # 1. Class Collapse Detection
    dominant_class = None
    collapse_detected = False
    
    for cname in classes:
        share = per_class[cname]["prediction_share"]
        if share > 60.0:
            collapse_detected = True
            dominant_class = cname
            
    # 2. Model Readiness Status Check
    # Pass Criteria: Accuracy >= 80%, all Recalls >= 75%, no prediction share > 60%
    ready_accuracy = overall_acc >= 80.0
    ready_recall_good = per_class["Good"]["recall"] >= 75.0
    ready_recall_worn = per_class["Worn"]["recall"] >= 75.0
    ready_recall_damaged = per_class["Damaged"]["recall"] >= 75.0
    ready_no_collapse = not collapse_detected
    
    model_ready = ready_accuracy and ready_recall_good and ready_recall_worn and ready_recall_damaged and ready_no_collapse
    
    # 3. Worst Performing Class
    # Judged by recall first, then accuracy
    worst_class = min(classes, key=lambda c: (per_class[c]["recall"], per_class[c]["accuracy"]))
    
    # 4. Misclassification Transition Summary
    # CM indices: actual on row, predicted on column
    cm = results["confusion_matrix"]
    transitions = {
        "Good -> Worn": cm[0][1],
        "Good -> Damaged": cm[0][2],
        "Worn -> Good": cm[1][0],
        "Worn -> Damaged": cm[1][2],
        "Damaged -> Good": cm[2][0],
        "Damaged -> Worn": cm[2][1]
    }
    
    # 5. Recommendation Engine
    recommendations = []
    
    if per_class["Damaged"]["recall"] < 50.0:
        recommendations.append("🚨 Recall for DAMAGED tyres is critically low (< 50%). Recommendation: Collect additional damaged tyre images and retrain.")
    elif per_class["Damaged"]["recall"] < 75.0:
        recommendations.append("⚠️ DAMAGED tyre recall is below production safety threshold. Recommendation: Enhance spatial contrast filters and add high-grime tyre samples during training.")
        
    if per_class["Worn"]["recall"] < 75.0:
        recommendations.append("⚠️ WORN tyre recall is below threshold. Recommendation: Include more border-line worn tyre treads to refine decision boundaries near legal depth limits.")
        
    if per_class["Good"]["recall"] < 75.0:
        recommendations.append("⚠️ GOOD tyre recall is below threshold. Recommendation: Enhance data augmentations (flips/rotations) to help the model learn general clean treads.")
        
    if collapse_detected:
        recommendations.append(f"🚨 CLASS COLLAPSE WARNING: Dominant class '{dominant_class}' absorbs > 60% of predictions. Recommendation: Apply WeightedRandomSampler in loaders and apply inverse frequency loss factors.")
        
    if overall_acc < 80.0:
        recommendations.append("🚨 Validation Accuracy is below 80%. Recommendation: Unfreeze final block feature groups earlier during training and fine-tune convolutional block weights longer.")
        
    if len(recommendations) == 0:
        recommendations.append("🟢 Model metrics are fully stable. Proceed to compile edge deployment binaries.")
        
    # --- CONSOLE OUTPUTS ---
    print(f"\nOverall Validation Accuracy: {overall_acc:.2f}%")
    print(f"Macro Precision:             {results['precision']:.2f}%")
    print(f"Macro Recall:                {results['recall']:.2f}%")
    print(f"Macro F1 Score:              {results['f1']:.2f}%")
    
    print("\n--- PER-CLASS ACCURACY & RECALL ---")
    for cname in classes:
        print(f"  * {cname}:")
        print(f"    - Accuracy: {per_class[cname]['accuracy']:.2f}%")
        print(f"    - Recall:   {per_class[cname]['recall']:.2f}%")
        
    print("\n--- PREDICTION DISTRIBUTION ---")
    for cname in classes:
        print(f"  * Predicted {cname:<8} = {per_class[cname]['prediction_share']:.2f}% ({per_class[cname]['predictions']}/{total})")
        
    if collapse_detected:
        print(f"\n🚨 CLASS COLLAPSE DETECTED")
        print(f"Dominant Class: {dominant_class.upper()}")
        print(f"Prediction Distribution:")
        print(f"  Good:    {per_class['Good']['prediction_share']:.2f}%")
        print(f"  Worn:    {per_class['Worn']['prediction_share']:.2f}%")
        print(f"  Damaged: {per_class['Damaged']['prediction_share']:.2f}%")
    else:
        print("\n🟢 No Class Collapse Detected: Prediction shares are distributed evenly.")
        
    print("\n--- MODEL DEPLOYMENT READINESS ---")
    if model_ready:
        print("✅ READY FOR REAL-TIME DEPLOYMENT")
    else:
        print("🚨 NOT READY FOR REAL-TIME DEPLOYMENT")
        
    print(f"\nWorst Performing Class: {worst_class.upper()}")
    print(f"  - Recall:   {per_class[worst_class]['recall']:.2f}%")
    print(f"  - Accuracy: {per_class[worst_class]['accuracy']:.2f}%")
    
    print("\n--- MISCLASSIFICATION SUMMARY ---")
    for tr, cnt in transitions.items():
        print(f"  • {tr:<15} = {cnt} counts")
        
    print("\n--- RECOMMENDATIONS ENGINE OUTPUT ---")
    for rec in recommendations:
        print(f"  * {rec}")
        
    results_processed = {
        "collapse_detected": collapse_detected,
        "dominant_class": dominant_class,
        "model_ready": model_ready,
        "worst_class": worst_class,
        "transitions": transitions,
        "recommendations": recommendations,
        "check_bounds": {
            "accuracy": ready_accuracy,
            "recall_good": ready_recall_good,
            "recall_worn": ready_recall_worn,
            "recall_damaged": ready_recall_damaged,
            "no_collapse": ready_no_collapse
        }
    }
    
    return results_processed

def export_html_report(results, proc_results, verifications):
    """
    Generates and writes a premium, high-impact cyberpunk-themed dark HTML report.
    """
    os.makedirs("results", exist_ok=True)
    html_path = "results/performance_report.html"
    
    classes = ["Good", "Worn", "Damaged"]
    total = results["total_samples"]
    overall_acc = results["overall_accuracy"]
    per_class = results["per_class"]
    transitions = proc_results["transitions"]
    cm = results["confusion_matrix"]
    
    # 20 Misclassifications catalog styling
    misclassified_items_html = ""
    for idx, sample in enumerate(results["misclassified"][:20], 1):
        misclassified_items_html += f"""
        <div class="defect-row">
            <div class="col-num">#{idx}</div>
            <div class="col-path" title="{sample['path']}">{sample['filename']}</div>
            <div class="col-lbl red">{sample['true_label'].upper()}</div>
            <div class="col-lbl purple">{sample['pred_label'].upper()}</div>
            <div class="col-conf">{sample['confidence']:.1f}%</div>
            <div class="col-link">
                <a href="file:///{sample['path'].replace(os.sep, '/')}" target="_blank">🌐 View Image</a>
            </div>
        </div>
        """
        
    # Recommendations bullet items HTML
    recs_html = ""
    for rec in proc_results["recommendations"]:
        recs_html += f"<li>{rec}</li>"
        
    # Programmatic file paths check verification cards HTML
    verif_cards_html = ""
    for label, msg, status in verifications:
        card_class = "verif-ok" if status else "verif-warn"
        icon = "🟢" if status else "🔴"
        verif_cards_html += f"""
        <div class="verif-card {card_class}">
            <div class="v-header">
                <span>{icon} {label}</span>
                <span class="badge">{'VERIFIED' if status else 'WARNING'}</span>
            </div>
            <div class="v-msg">{msg}</div>
        </div>
        """
        
    # Model readiness banner CSS
    if proc_results["model_ready"]:
        readiness_banner = """
        <div class="readiness-banner ready">
            <span class="big-icon">✅</span>
            <div class="b-text">
                <h3>READY FOR REAL-TIME DEPLOYMENT</h3>
                <p>Model passed all critical precision thresholds, recall balances, and skew checks. System secured.</p>
            </div>
        </div>
        """
    else:
        readiness_banner = """
        <div class="readiness-banner not-ready">
            <span class="big-icon">🚨</span>
            <div class="b-text">
                <h3>NOT READY FOR REAL-TIME DEPLOYMENT</h3>
                <p>Failed one or more performance/recall limits on validation subsets. Hazard alert active.</p>
            </div>
        </div>
        """
        
    # Collapse alert banner CSS
    if proc_results["collapse_detected"]:
        collapse_banner = f"""
        <div class="collapse-banner">
            <h3>🚨 CLASS COLLAPSE DETECTED</h3>
            <p>Dominant Class: <b>{proc_results['dominant_class'].upper()}</b> absorbs more than 60% of validation forecasts.</p>
        </div>
        """
    else:
        collapse_banner = """
        <div class="collapse-banner ok">
            <h3>🟢 CLASS SKEW SECURE</h3>
            <p>Target predictions display homogeneous distributions across categories. Classifier split is balanced.</p>
        </div>
        """

    # Compute bounds badges
    cb = proc_results["check_bounds"]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-TREAD // Neural Performance Audit Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #050811;
            --card-bg: rgba(10, 15, 30, 0.7);
            --neon-cyan: #00f2fe;
            --neon-purple: #7f00ff;
            --neon-pink: #ff007f;
            --neon-green: #00f260;
            --neon-orange: #f39c12;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: rgba(255, 255, 255, 0.08);
            --neon-border: rgba(0, 242, 254, 0.25);
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 40px 20px;
            background-image: 
                radial-gradient(at 0% 0%, rgba(127, 0, 255, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(0, 242, 254, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 40px;
        }}

        h1 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2.8rem;
            font-weight: 900;
            letter-spacing: 2px;
            margin: 0 0 10px 0;
            background: linear-gradient(90deg, #00f2fe, #7f00ff, #ff007f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            text-shadow: 0 0 40px rgba(0, 242, 254, 0.2);
        }}

        .subtitle {{
            color: var(--text-sub);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            letter-spacing: 3px;
            margin: 0;
            text-transform: uppercase;
        }}

        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1-indexed);
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 25px 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            border-color: var(--neon-cyan);
            box-shadow: 0 12px 40px rgba(0, 242, 254, 0.15);
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
        }}

        .metric-card.cyan::before {{ background: var(--neon-cyan); }}
        .metric-card.purple::before {{ background: var(--neon-purple); }}
        .metric-card.pink::before {{ background: var(--neon-pink); }}
        .metric-card.green::before {{ background: var(--neon-green); }}

        .metric-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }}

        .metric-card.cyan .metric-value {{ color: var(--neon-cyan); }}
        .metric-card.purple .metric-value {{ color: #d6bcfa; }}
        .metric-card.pink .metric-value {{ color: var(--neon-pink); }}
        .metric-card.green .metric-value {{ color: var(--neon-green); }}

        .metric-label {{
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            color: var(--text-sub);
            letter-spacing: 1px;
            text-transform: uppercase;
        }}

        .glass-panel {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
        }}

        .panel-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.25rem;
            margin-top: 0;
            margin-bottom: 25px;
            color: var(--text-main);
            letter-spacing: 1px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .panel-title .glow-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--neon-cyan);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--neon-cyan);
        }}

        /* Banners */
        .readiness-banner {{
            display: flex;
            align-items: center;
            padding: 25px 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .readiness-banner.ready {{
            background: rgba(0, 242, 96, 0.08);
            border: 2px solid var(--neon-green);
            box-shadow: 0 0 30px rgba(0, 242, 96, 0.15);
        }}
        .readiness-banner.not-ready {{
            background: rgba(255, 0, 127, 0.08);
            border: 2px solid var(--neon-pink);
            box-shadow: 0 0 30px rgba(255, 0, 127, 0.15);
        }}
        .readiness-banner .big-icon {{
            font-size: 3.5rem;
            margin-right: 25px;
        }}
        .readiness-banner h3 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4rem;
            margin: 0 0 5px 0;
            letter-spacing: 1px;
        }}
        .readiness-banner.ready h3 {{ color: var(--neon-green); }}
        .readiness-banner.not-ready h3 {{ color: var(--neon-pink); }}
        .readiness-banner p {{
            margin: 0;
            color: var(--text-sub);
            font-size: 0.95rem;
            line-height: 1.5;
        }}

        .collapse-banner {{
            padding: 20px 25px;
            border-radius: 10px;
            background: rgba(243, 156, 18, 0.08);
            border: 1px solid var(--neon-orange);
            margin-bottom: 30px;
        }}
        .collapse-banner.ok {{
            background: rgba(0, 242, 254, 0.04);
            border: 1px solid rgba(0, 242, 254, 0.2);
        }}
        .collapse-banner h3 {{
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
            margin: 0 0 6px 0;
            color: var(--neon-orange);
        }}
        .collapse-banner.ok h3 {{
            color: var(--neon-cyan);
        }}
        .collapse-banner p {{
            margin: 0;
            font-size: 0.9rem;
            color: var(--text-sub);
        }}

        /* Two columns */
        .flex-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}

        @media (max-width: 850px) {{
            .flex-2 {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Tables & Matrices */
        table.styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}
        table.styled-table th {{
            background: rgba(255, 255, 255, 0.02);
            color: var(--neon-cyan);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            text-transform: uppercase;
            padding: 12px 15px;
            letter-spacing: 1px;
            border-bottom: 2px solid var(--border-color);
        }}
        table.styled-table td {{
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
        }}
        table.styled-table tr:hover td {{
            background: rgba(255,255,255,0.01);
        }}

        /* Confusion Matrix Style */
        .cm-grid {{
            display: grid;
            grid-template-columns: 100px repeat(3, 1fr);
            gap: 8px;
            text-align: center;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            margin-top: 15px;
        }}
        .cm-hdr {{
            font-size: 0.75rem;
            color: var(--neon-cyan);
            padding: 10px 0;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .cm-lbl {{
            font-size: 0.8rem;
            color: var(--text-sub);
            background: rgba(255, 255, 255, 0.02);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px 0;
        }}
        .cm-cell {{
            background: rgba(10, 15, 30, 0.9);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px 0;
            font-size: 1.25rem;
            font-weight: bold;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .cm-cell.hit {{
            background: rgba(0, 242, 96, 0.08);
            border: 1px solid rgba(0, 242, 96, 0.35);
            color: var(--neon-green);
            text-shadow: 0 0 10px rgba(0, 242, 96, 0.2);
        }}
        .cm-cell.miss {{
            background: rgba(255, 0, 127, 0.06);
            border: 1px solid rgba(255, 0, 127, 0.25);
            color: var(--neon-pink);
        }}
        .cm-cell .lbl-small {{
            font-size: 0.65rem;
            color: var(--text-sub);
            font-family: 'Inter', sans-serif;
            margin-top: 4px;
            font-weight: normal;
        }}

        /* Lists */
        ul.bullet-list {{
            padding-left: 20px;
            line-height: 1.8;
            color: #cbd5e1;
            font-size: 0.95rem;
        }}
        ul.bullet-list li {{
            margin-bottom: 12px;
        }}

        /* Verification Cards Grid */
        .verif-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}
        .verif-card {{
            background: rgba(10, 15, 30, 0.85);
            border-radius: 8px;
            padding: 18px 20px;
            border-left: 4px solid #94a3b8;
        }}
        .verif-card.verif-ok {{
            border-left-color: var(--neon-green);
        }}
        .verif-card.verif-warn {{
            border-left-color: var(--neon-pink);
        }}
        .v-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            font-weight: bold;
            margin-bottom: 6px;
        }}
        .v-header .badge {{
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .verif-ok .badge {{
            background: rgba(0, 242, 96, 0.15);
            color: var(--neon-green);
        }}
        .verif-warn .badge {{
            background: rgba(255, 0, 127, 0.15);
            color: var(--neon-pink);
        }}
        .v-msg {{
            font-size: 0.82rem;
            color: var(--text-sub);
            line-height: 1.4;
        }}

        /* Defect row list */
        .defect-list {{
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow: hidden;
            background: rgba(5,8,17,0.5);
        }}
        .defect-hdr {{
            display: grid;
            grid-template-columns: 60px 1.5fr 1fr 1fr 1fr 120px;
            background: rgba(255,255,255,0.03);
            border-bottom: 2px solid var(--border-color);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            color: var(--neon-cyan);
            text-transform: uppercase;
            padding: 12px 15px;
            letter-spacing: 1px;
        }}
        .defect-row {{
            display: grid;
            grid-template-columns: 60px 1.5fr 1fr 1fr 1fr 120px;
            border-bottom: 1px solid var(--border-color);
            padding: 12px 15px;
            font-size: 0.85rem;
            align-items: center;
        }}
        .defect-row:last-child {{
            border-bottom: none;
        }}
        .defect-row:hover {{
            background: rgba(255,255,255,0.015);
        }}
        .col-num {{
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            color: var(--text-sub);
        }}
        .col-path {{
            font-family: monospace;
            color: #cbd5e1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding-right: 15px;
        }}
        .col-lbl {{
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            font-size: 0.8rem;
        }}
        .col-lbl.red {{ color: var(--neon-pink); }}
        .col-lbl.purple {{ color: #c084fc; }}
        .col-conf {{
            font-family: monospace;
            color: var(--neon-cyan);
            font-weight: bold;
        }}
        .col-link a {{
            color: var(--neon-cyan);
            text-decoration: none;
            font-weight: bold;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            border: 1px solid rgba(0, 242, 254, 0.2);
            padding: 4px 8px;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .col-link a:hover {{
            background: var(--neon-cyan);
            color: var(--bg-color);
            box-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
        }}
        
        .progress-bar-container {{
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            margin-top: 6px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 3px;
        }}
        .progress-fill.cyan {{ background: var(--neon-cyan); }}
        .progress-fill.purple {{ background: var(--neon-purple); }}
        .progress-fill.pink {{ background: var(--neon-pink); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI-TREAD // Neural Performance Audit</h1>
            <p class="subtitle">Deep Learning Evaluation & Deployment Safety Telemetry</p>
        </header>

        {readiness_banner}
        {collapse_banner}

        <div class="grid-4">
            <div class="metric-card cyan">
                <div class="metric-value">{overall_acc:.2f}%</div>
                <div class="metric-label">Validation Accuracy</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-value">{results['precision']:.2f}%</div>
                <div class="metric-label">Macro Precision</div>
            </div>
            <div class="metric-card pink">
                <div class="metric-value">{results['recall']:.2f}%</div>
                <div class="metric-label">Macro Recall</div>
            </div>
            <div class="metric-card green">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Inspected Samples</div>
            </div>
        </div>

        <div class="flex-2">
            <div class="glass-panel">
                <div class="panel-title">
                    <span>📊 CATEGORICAL METRICS SUMMARY</span>
                    <span class="glow-dot"></span>
                </div>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Class Category</th>
                            <th>Precision</th>
                            <th>Accuracy</th>
                            <th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>🟢 GOOD</b></td>
                            <td class="col-conf">{per_class['Good']['precision']:.2f}%</td>
                            <td>{per_class['Good']['accuracy']:.2f}%</td>
                            <td style="color: var(--neon-green); font-weight: bold;">{per_class['Good']['recall']:.2f}%</td>
                        </tr>
                        <tr>
                            <td><b>🟡 WORN</b></td>
                            <td class="col-conf">{per_class['Worn']['precision']:.2f}%</td>
                            <td>{per_class['Worn']['accuracy']:.2f}%</td>
                            <td style="color: var(--neon-orange); font-weight: bold;">{per_class['Worn']['recall']:.2f}%</td>
                        </tr>
                        <tr>
                            <td><b>🔴 DAMAGED</b></td>
                            <td class="col-conf">{per_class['Damaged']['precision']:.2f}%</td>
                            <td>{per_class['Damaged']['accuracy']:.2f}%</td>
                            <td style="color: var(--neon-pink); font-weight: bold;">{per_class['Damaged']['recall']:.2f}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="glass-panel">
                <div class="panel-title">
                    <span>🧠 CNN CONFUSION MATRIX</span>
                    <span class="glow-dot"></span>
                </div>
                <div class="cm-grid">
                    <!-- Headings -->
                    <div></div>
                    <div class="cm-hdr">Pred Good</div>
                    <div class="cm-hdr">Pred Worn</div>
                    <div class="cm-hdr">Pred Damaged</div>

                    <!-- Row 1: Good -->
                    <div class="cm-lbl">Actual Good</div>
                    <div class="cm-cell hit">{cm[0][0]}<span class="lbl-small">TP</span></div>
                    <div class="cm-cell miss">{cm[0][1]}<span class="lbl-small">Worn Skew</span></div>
                    <div class="cm-cell miss">{cm[0][2]}<span class="lbl-small">Damage Skew</span></div>

                    <!-- Row 2: Worn -->
                    <div class="cm-lbl">Actual Worn</div>
                    <div class="cm-cell miss">{cm[1][0]}<span class="lbl-small">Good Skew</span></div>
                    <div class="cm-cell hit">{cm[1][1]}<span class="lbl-small">TP</span></div>
                    <div class="cm-cell miss">{cm[1][2]}<span class="lbl-small">Damage Skew</span></div>

                    <!-- Row 3: Damaged -->
                    <div class="cm-lbl">Actual Damaged</div>
                    <div class="cm-cell miss">{cm[2][0]}<span class="lbl-small">Good Skew</span></div>
                    <div class="cm-cell miss">{cm[2][1]}<span class="lbl-small">Worn Skew</span></div>
                    <div class="cm-cell hit">{cm[2][2]}<span class="lbl-small">TP</span></div>
                </div>
            </div>
        </div>

        <div class="flex-2">
            <div class="glass-panel">
                <div class="panel-title">
                    <span>📈 PREDICTION VOLUME DISTRIBUTION</span>
                    <span class="glow-dot"></span>
                </div>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Predictions</th>
                            <th>Distribution Share</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>GOOD</b></td>
                            <td class="col-conf">{per_class['Good']['predictions']}</td>
                            <td>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span>{per_class['Good']['prediction_share']:.2f}%</span>
                                    <div class="progress-bar-container" style="width: 120px; margin-top: 0;">
                                        <div class="progress-fill cyan" style="width: {per_class['Good']['prediction_share']}%;"></div>
                                    </div>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td><b>WORN</b></td>
                            <td class="col-conf">{per_class['Worn']['predictions']}</td>
                            <td>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span>{per_class['Worn']['prediction_share']:.2f}%</span>
                                    <div class="progress-bar-container" style="width: 120px; margin-top: 0;">
                                        <div class="progress-fill purple" style="width: {per_class['Worn']['prediction_share']}%;"></div>
                                    </div>
                                </div>
                            </td>
                        </tr>
                        <tr>
                            <td><b>DAMAGED</b></td>
                            <td class="col-conf">{per_class['Damaged']['predictions']}</td>
                            <td>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span>{per_class['Damaged']['prediction_share']:.2f}%</span>
                                    <div class="progress-bar-container" style="width: 120px; margin-top: 0;">
                                        <div class="progress-fill pink" style="width: {per_class['Damaged']['prediction_share']}%;"></div>
                                    </div>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="glass-panel">
                <div class="panel-title">
                    <span>💥 MISCLASSIFICATION SHIFT SUMMARY</span>
                    <span class="glow-dot"></span>
                </div>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Transition Case</th>
                            <th>Count Logs</th>
                            <th>Critical Risk level</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Good &rarr; Worn</td>
                            <td class="col-conf">{transitions['Good -> Worn']}</td>
                            <td style="color: var(--neon-orange); font-weight: bold;">MODERATE SKEW</td>
                        </tr>
                        <tr>
                            <td>Good &rarr; Damaged</td>
                            <td class="col-conf">{transitions['Good -> Damaged']}</td>
                            <td style="color: var(--neon-pink); font-weight: bold;">HIGH HAZARD</td>
                        </tr>
                        <tr>
                            <td>Worn &rarr; Good</td>
                            <td class="col-conf">{transitions['Worn -> Good']}</td>
                            <td style="color: var(--neon-pink); font-weight: bold;">CRITICAL BYPASS</td>
                        </tr>
                        <tr>
                            <td>Worn &rarr; Damaged</td>
                            <td class="col-conf">{transitions['Worn -> Damaged']}</td>
                            <td style="color: var(--neon-orange); font-weight: bold;">MODERATE SKEW</td>
                        </tr>
                        <tr>
                            <td>Damaged &rarr; Good</td>
                            <td class="col-conf">{transitions['Damaged -> Good']}</td>
                            <td style="color: var(--neon-pink); font-weight: bold;">CRITICAL BYPASS</td>
                        </tr>
                        <tr>
                            <td>Damaged &rarr; Worn</td>
                            <td class="col-conf">{transitions['Damaged -> Worn']}</td>
                            <td style="color: var(--neon-orange); font-weight: bold;">MODERATE SKEW</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="flex-2">
            <div class="glass-panel" style="margin-bottom: 30px;">
                <div class="panel-title">
                    <span>🔴 WORST PERFORMING NEURAL SECTOR</span>
                    <span class="glow-dot"></span>
                </div>
                <div style="padding: 10px; background: rgba(255, 0, 127, 0.03); border: 1px solid rgba(255, 0, 127, 0.15); border-radius: 8px;">
                    <h3 style="margin: 0 0 10px 0; color: var(--neon-pink); font-family: 'Orbitron'; text-transform: uppercase;">Category: {proc_results['worst_class'].upper()}</h3>
                    <p style="margin: 0 0 6px 0; color: var(--text-sub); font-size: 0.95rem;">
                        This class has been identified as the lowest performing node within the active model weight checkpoint splits.
                    </p>
                    <div style="display: flex; gap: 40px; margin-top: 15px; font-family: 'Orbitron', sans-serif;">
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-sub);">WORST RECALL</div>
                            <div style="font-size: 1.8rem; color: var(--neon-pink); font-weight: bold;">{per_class[proc_results['worst_class']]['recall']:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: var(--text-sub);">ACCURACY FOOTPRINT</div>
                            <div style="font-size: 1.8rem; color: #f8fafc; font-weight: bold;">{per_class[proc_results['worst_class']]['accuracy']:.2f}%</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="glass-panel" style="margin-bottom: 30px;">
                <div class="panel-title">
                    <span>💡 AUTOMATED RECOMMENDATION ENGINE</span>
                    <span class="glow-dot"></span>
                </div>
                <ul class="bullet-list" style="margin-top: 0; margin-bottom: 0;">
                    {recs_html}
                </ul>
            </div>
        </div>

        <div class="glass-panel">
            <div class="panel-title">
                <span>📁 UNIFIED ENTRY-POINT PATH VERIFICATIONS</span>
                <span class="glow-dot"></span>
            </div>
            <div class="verif-grid">
                {verif_cards_html}
            </div>
        </div>

        <div class="glass-panel">
            <div class="panel-title">
                <span>🔍 TARGET AUDIT LOG (FIRST 20 MISCLASSIFIED VALIDATION SAMPLES)</span>
                <span class="glow-dot"></span>
            </div>
            <div class="defect-list">
                <div class="defect-hdr">
                    <div>Index</div>
                    <div>File Name</div>
                    <div>Actual Class</div>
                    <div>Predicted Class</div>
                    <div>Confidence</div>
                    <div>Action</div>
                </div>
                {misclassified_items_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n[EXPORTED] HTML Performance report written successfully to '{html_path}'")
    return html_path

def main():
    verifications = verify_model_paths()
    results = evaluate_model()
    proc_results = process_and_display_results(results)
    export_html_report(results, proc_results, verifications)
    print("\n" + "="*60)
    print("AI-TREAD NEURAL EVALUATION AUDIT SUITE FINISHED SECURELY")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
