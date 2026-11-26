import sys
import numpy as np
import collections
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# Add project root to path for imports
PROJECT_ROOT = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model')
sys.path.insert(0, str(PROJECT_ROOT))

from mr_tae_fusion.data.distance_signals import generate_noisy_signal, DistanceSignalConfig

def load_mat_file(filepath):
    try:
        import scipy.io as sio
        return sio.loadmat(str(filepath))
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(filepath))

def main():
    DATA_DIR = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Datset from Q.lin')
    OUT_DIR = DATA_DIR / 'processed'
    OUT_DIR.mkdir(exist_ok=True)
    
    files = [
        ('1.0mm', 'data1.0mm.mat', 0),
        ('1.8mm', 'data1.8mm.mat', 1),
        ('2.0mm', 'data2.0mm.mat', 2),
        ('2.5mm', 'data2.5mm.mat', 3)
    ]
    
    config = DistanceSignalConfig()
    rng = np.random.default_rng(42)
    
    X_all = []
    y_all = []
    
    print("Loading real data...")
    for label, filename, class_idx in files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist.")
            continue
            
        mat = load_mat_file(filepath)
        data = None
        for key, value in mat.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                data = value
                break
                
        if data is None:
            continue
            
        # Preprocessing matching compare_methods.py logic
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] > data.shape[1]:
            data = data.T
            
        # Standardize strictly to 2001 length and normalize the amplitudes
        processed = []
        # Process ALL samples instead of limiting for the 10/90 split
        for i in range(data.shape[0]):
            sig = data[i]
            if len(sig) >= 2001:
                sig = sig[:2001]
            else:
                sig = np.pad(sig, (0, 2001 - len(sig)))
            
            # Normalize pure real data amplitude
            max_val = np.max(np.abs(sig))
            if max_val > 0:
                sig = sig / max_val
            
            processed.append(sig)
            
        X_all.append(np.array(processed))
        y_all.extend([class_idx] * len(processed))
        print(f"Loaded {len(processed)} samples for {label}")
        
    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    
    print(f"\nTotal real samples: {len(y_all)}")
    
    # Stratified 10/90 Split
    X_train_clean, X_test_clean, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.90, random_state=42, stratify=y_all
    )
    
    print(f"Split: 10% Train ({len(y_train)}), 90% Test ({len(y_test)})")
    
    print("\nAugmenting 10% Training Data with AE Noise (Reflection + AEPD + AWGN)...")
    X_train_noisy = []
    train_snrs = []
    
    for i in range(len(X_train_clean)):
        clean_sig = X_train_clean[i]
        class_idx = y_train[i]
        
        # Apply synthetic noisy signal layer to pure data
        # Target a random SNR between -10 and +10 to ensure robustness
        target_snr = rng.uniform(-10, 10)
        
        noisy_sig, actual_snr = generate_noisy_signal(
            clean_sig,
            target_snr_db=target_snr,
            distance_class=class_idx,
            config=config,
            rng=rng
        )
        
        X_train_noisy.append(noisy_sig)
        train_snrs.append(actual_snr)
        
    X_train_noisy = np.array(X_train_noisy)
    
    # Save processed splits
    print(f"Saving splits to {OUT_DIR} ...")
    np.savez_compressed(
        OUT_DIR / "real_augmented_train.npz",
        clean=X_train_clean,
        noisy=X_train_noisy,
        labels=y_train,
        snrs=np.array(train_snrs)
    )
    
    np.savez_compressed(
        OUT_DIR / "real_test_clean.npz",
        clean=X_test_clean,      # The test bench gets raw clean signals, optionally noisy versions are evaluated dynamically
        labels=y_test
    )
    
    print("Done! Data prepared successfully.")

if __name__ == "__main__":
    main()
