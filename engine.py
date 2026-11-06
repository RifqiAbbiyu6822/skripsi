import math
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

def train_one_epoch(model, data_loader, optimizer, criterion, device, scaler=None, grad_clip=None, ema_model=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(data_loader, desc="Training", leave=False, file=sys.stdout)
    
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = inputs.size(0)
        optimizer.zero_grad()
        
        # Pakai format autocast terbaru dari PyTorch biar ga ada warning
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        loss_val = loss.item()
        # Guard: hindari loss NaN/Inf masuk ke model
        if not math.isfinite(loss_val):
            print(f"\n[WARNING] Loss is {loss_val}, skipping batch.")
            continue
            
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Gradient clipping via scaler
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        if ema_model is not None:
            ema_model.update(model)
            
        total_loss += loss_val * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({'loss': f"{loss_val:.4f}"})
        
    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, target_classes, threshold=0.5, return_probs=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(data_loader, desc="Evaluating", leave=False, file=sys.stdout)
    
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        loss_val = loss.item()
        if math.isfinite(loss_val):
            total_loss += loss_val * batch_size
            total_samples += batch_size
            
        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Binarize prediksi sesuai threshold buat ngitung F1
    bin_preds = (all_preds >= threshold).astype(int)
    
    # Hitung metrik
    macro_f1 = f1_score(all_targets, bin_preds, average='macro', zero_division=0)
    
    try:
        macro_auroc = roc_auc_score(all_targets, all_preds, average='macro')
    except ValueError:
        # Jaga-jaga kalau ada label kelas yang cuma negatif di split val
        macro_auroc = 0.0
        print("\n[WARNING] ROC AUC ValueError: Kelas tidak punya variasi label positif di batch ini.")
        
    metrics_dict = {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc
    }
    
    # Ekstrak F1 per kelas biar f1_HYP bisa dibaca sama train_report lo
    per_class_f1 = f1_score(all_targets, bin_preds, average=None, zero_division=0)
    for i, class_name in enumerate(target_classes):
        metrics_dict[f"f1_{class_name}"] = per_class_f1[i]
        
    if return_probs:
        return metrics_dict, all_preds, all_targets
        
    return metrics_dict