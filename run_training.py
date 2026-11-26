"""
Training Script - Distance-Based PD Classification
Uses myenv with CUDA for GPU acceleration
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add project to path
PROJECT_ROOT = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model')
sys.path.insert(0, str(PROJECT_ROOT))

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("="*70)
print("DISTANCE-BASED PD CLASSIFICATION")
print("="*70)

# Import modules
from mr_tae_fusion.data.distance_signals import (
    generate_batch, 
    DistanceSignalConfig
)
from mr_tae_fusion.models.distance_classifier import DistanceClassifier

# Configuration
config = DistanceSignalConfig()

# Generate training data
print("\n1. GENERATING DATA...")
print("-"*70)

train_data = generate_batch(
    batch_size=200,  # 200 samples per class = 800 total
    config=config,
    snr_range=(-10, 5),
    seed=42
)
print(f"Training: {len(train_data['labels'])} samples")

val_data = generate_batch(
    batch_size=50,
    config=config,
    snr_range=(-10, 5),
    seed=123
)
print(f"Validation: {len(val_data['labels'])} samples")

test_data = generate_batch(
    batch_size=50,
    config=config,
    snr_range=(-15, 0),  # Harder test
    seed=456
)
print(f"Test: {len(test_data['labels'])} samples")

print(f"Class distribution: {np.bincount(train_data['labels'])}")

# Create dataloaders
def create_dataloader(data_dict, batch_size=32, shuffle=True):
    X = torch.tensor(data_dict['noisy'], dtype=torch.float32).unsqueeze(1)
    X_clean = torch.tensor(data_dict['clean'], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data_dict['labels'], dtype=torch.long)
    dataset = TensorDataset(X, X_clean, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(train_data, shuffle=True)
val_loader = create_dataloader(val_data, shuffle=False)
test_loader = create_dataloader(test_data, shuffle=False)

# Create model
print("\n2. CREATING MODEL...")
print("-"*70)

classifier = DistanceClassifier(
    in_channels=1,
    signal_length=2001,
    num_classes=4,
    hidden_channels=[32, 64, 128]
).to(device)

total_params = sum(p.numel() for p in classifier.parameters())
print(f"Model parameters: {total_params:,}")

# Training functions
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for noisy, clean, labels in loader:
        X = clean.to(device)  # Train on clean first
        y = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, use_noisy=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for noisy, clean, labels in loader:
            X = noisy.to(device) if use_noisy else clean.to(device)
            y = labels.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels

# Training
print("\n3. TRAINING...")
print("-"*70)

NUM_EPOCHS = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0
best_state = None

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = evaluate(classifier, val_loader, criterion)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = classifier.state_dict().copy()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
classifier.load_state_dict(best_state)

# Evaluation
print("\n4. EVALUATION (Simulated Data)")
print("-"*70)

_, clean_acc, clean_preds, clean_labels = evaluate(classifier, test_loader, criterion, use_noisy=False)
_, noisy_acc, noisy_preds, noisy_labels = evaluate(classifier, test_loader, criterion, use_noisy=True)

print(f"Accuracy on CLEAN signals: {clean_acc:.2f}%")
print(f"Accuracy on NOISY signals: {noisy_acc:.2f}%")
print(f"Performance drop: {clean_acc - noisy_acc:.2f}%")

# Load and test on real data
print("\n5. REAL-WORLD DATA EVALUATION")
print("-"*70)

def load_mat_file(filepath):
    try:
        import scipy.io as sio
        return sio.loadmat(str(filepath))
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(filepath))

DATA_DIR = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Datset from Q.lin')
files = [
    ('1.0mm', 'data1.0mm.mat', 0),
    ('1.8mm', 'data1.8mm.mat', 1),
    ('2.0mm', 'data2.0mm.mat', 2),
    ('2.5mm', 'data2.5mm.mat', 3)
]

all_preds, all_labels = [], []
class_names = ['1.0mm', '1.8mm', '2.0mm', '2.5mm']

for label, filename, class_idx in files:
    filepath = DATA_DIR / filename
    if not filepath.exists():
        continue
    
    mat = load_mat_file(filepath)
    data = None
    for key, value in mat.items():
        if not key.startswith('__') and isinstance(value, np.ndarray):
            data = value
            break
    
    if data is None:
        continue
    
    # Preprocess
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    # Pad/truncate to 2001
    processed = []
    for i in range(min(50, data.shape[0])):  # Limit to 50 samples
        sig = data[i]
        if len(sig) >= 2001:
            sig = sig[:2001]
        else:
            sig = np.pad(sig, (0, 2001 - len(sig)))
        # Normalize
        if np.max(np.abs(sig)) > 0:
            sig = sig / np.max(np.abs(sig))
        processed.append(sig)
    
    processed = np.array(processed)
    X = torch.tensor(processed, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        preds = classifier.predict(X).cpu().numpy()
    
    correct = np.sum(preds == class_idx)
    acc = 100.0 * correct / len(preds)
    
    all_preds.extend(preds)
    all_labels.extend([class_idx] * len(preds))
    
    pred_counts = np.bincount(preds, minlength=4)
    print(f"{label}: Accuracy={acc:.1f}% | Predictions: " +
          ", ".join([f"{n}={c}" for n, c in zip(class_names, pred_counts)]))

if len(all_preds) > 0:
    overall_acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nOVERALL ACCURACY: {overall_acc:.2f}%")

# Save model and plots
print("\n6. SAVING RESULTS...")
print("-"*70)

save_dir = PROJECT_ROOT / 'mr_tae_fusion' / 'checkpoints'
save_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': classifier.state_dict(),
    'history': history,
    'config': {'signal_length': 2001, 'num_classes': 4}
}, save_dir / 'distance_classifier.pth')
print(f"Model saved: {save_dir / 'distance_classifier.pth'}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Training Loss')

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].set_title('Training Accuracy')

plt.tight_layout()
plt.savefig(Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13') / 'training_curves.png', dpi=150)
print(f"Training curves saved: training_curves.png")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
