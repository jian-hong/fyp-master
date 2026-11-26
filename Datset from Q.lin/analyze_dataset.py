"""
Comprehensive Analysis of Q.Lin PD-induced AE Signal Dataset
=============================================================
This script verifies the claims from Lin et al. (2022) about the dataset:
- 200 seconds of continuous collection per defect model
- 4 defect types (copper beads: 1.0, 1.8, 2.0, 2.5 mm)
- Signals segmented into 1-second windows
- 800 total vectors (200 samples × 4 fault types)
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

# Dataset files
DATA_FILES = {
    '1.0mm': 'data1.0mm.mat',
    '1.8mm': 'data1.8mm.mat',
    '2.0mm': 'data2.0mm.mat',
    '2.5mm': 'data2.5mm.mat'
}

def analyze_mat_file(filepath, label):
    """Analyze a single .mat file and return statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {label} ({os.path.basename(filepath)})")
    print(f"{'='*60}")
    
    # Load the file
    mat_data = sio.loadmat(filepath)
    
    # Print all keys
    print(f"Keys in file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
    
    # Find the data key (exclude metadata keys)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    results = {}
    
    for key in data_keys:
        data = mat_data[key]
        print(f"\n--- Variable: '{key}' ---")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Total samples: {data.size}")
        
        # Flatten for analysis
        signal = data.flatten()
        
        print(f"  Min value: {np.min(signal):.6f}")
        print(f"  Max value: {np.max(signal):.6f}")
        print(f"  Mean value: {np.mean(signal):.6f}")
        print(f"  Std deviation: {np.std(signal):.6f}")
        
        results[key] = {
            'shape': data.shape,
            'total_samples': data.size,
            'signal': signal
        }
    
    return results

def estimate_sampling_parameters(total_samples, expected_duration_sec=200):
    """Estimate sampling rate and duration."""
    # Calculate sampling rate assuming 200 seconds
    estimated_fs = total_samples / expected_duration_sec
    
    # Check common sampling rates
    common_rates = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    closest_rate = min(common_rates, key=lambda x: abs(x - estimated_fs))
    
    return {
        'estimated_fs': estimated_fs,
        'closest_common_fs': closest_rate,
        'duration_at_estimated': total_samples / estimated_fs if estimated_fs > 0 else 0,
        'duration_at_closest': total_samples / closest_rate if closest_rate > 0 else 0
    }

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("="*70)
    print("Q.LIN PD-INDUCED AE SIGNAL DATASET ANALYSIS")
    print("Verifying claims from Lin et al. (2022)")
    print("="*70)
    
    all_results = {}
    
    # Analyze each file
    for label, filename in DATA_FILES.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            all_results[label] = analyze_mat_file(filepath, label)
        else:
            print(f"\n[WARNING] File not found: {filename}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: VERIFICATION OF PAPER CLAIMS")
    print("="*70)
    
    print("\n[DATA] CLAIMED vs ACTUAL DATA:")
    print("-" * 50)
    print("Paper claims:")
    print("  - 200 seconds continuous collection per defect")
    print("  - 4 defect types (1.0, 1.8, 2.0, 2.5 mm copper beads)")
    print("  - 1-second windows → 200 samples per defect")
    print("  - Total: 800 vectors (200 × 4)")
    print("-" * 50)
    
    print("\n[RESULTS] ACTUAL DATA FROM FILES:")
    
    total_vectors = 0
    
    for label, results in all_results.items():
        for key, data in results.items():
            total_samples = data['total_samples']
            params = estimate_sampling_parameters(total_samples)
            
            print(f"\n  {label} ({key}):")
            print(f"    Total samples: {total_samples:,}")
            print(f"    Estimated sampling rate: {params['estimated_fs']:,.0f} Hz")
            print(f"    Closest common rate: {params['closest_common_fs']:,} Hz")
            print(f"    Duration at {params['closest_common_fs']:,} Hz: {params['duration_at_closest']:.2f} seconds")
            
            # Estimate 1-second windows
            samples_per_window = params['closest_common_fs']
            num_windows = total_samples / samples_per_window
            print(f"    1-second windows (vectors): {num_windows:.0f}")
            total_vectors += num_windows
    
    print(f"\n[TOTAL] TOTAL VECTORS (all files): {total_vectors:.0f}")
    print(f"   Paper claims: 800 vectors")
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (label, filename) in enumerate(DATA_FILES.items()):
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath) and label in all_results:
            results = all_results[label]
            for key, data in results.items():
                signal = data['signal']
                
                # Show first 1 million samples for clarity (or all if less)
                display_samples = min(len(signal), 1000000)
                
                # Estimate time axis
                total_samples = len(signal)
                params = estimate_sampling_parameters(total_samples)
                fs = params['closest_common_fs']
                time_axis = np.arange(display_samples) / fs
                
                axes[idx].plot(time_axis, signal[:display_samples], linewidth=0.5)
                axes[idx].set_title(f'{label} - {key}\n({total_samples:,} samples, ~{total_samples/fs:.1f}s at {fs/1000:.0f}kHz)')
                axes[idx].set_xlabel('Time (seconds)')
                axes[idx].set_ylabel('Amplitude')
                axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Q.Lin PD-Induced AE Signals - Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'dataset_overview.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: dataset_overview.png")
    
    # Second figure: Zoom into 1-second segments
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    
    for idx, (label, filename) in enumerate(DATA_FILES.items()):
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath) and label in all_results:
            results = all_results[label]
            for key, data in results.items():
                signal = data['signal']
                total_samples = len(signal)
                params = estimate_sampling_parameters(total_samples)
                fs = params['closest_common_fs']
                
                # Show first 1-second window
                samples_1sec = int(fs)
                if len(signal) >= samples_1sec:
                    segment = signal[:samples_1sec]
                    time_axis = np.arange(samples_1sec) / fs * 1000  # in ms
                    
                    axes2[idx].plot(time_axis, segment, linewidth=0.5)
                    axes2[idx].set_title(f'{label} - First 1-Second Window\n({samples_1sec:,} samples)')
                    axes2[idx].set_xlabel('Time (ms)')
                    axes2[idx].set_ylabel('Amplitude')
                    axes2[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Q.Lin PD-Induced AE Signals - 1-Second Windows (Verification)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'one_second_windows.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: one_second_windows.png")
    
    print("\n[DONE] Analysis complete!")

if __name__ == "__main__":
    main()
