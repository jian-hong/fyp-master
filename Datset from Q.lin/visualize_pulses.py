"""
Enhanced Visualization of Q.Lin PD-Induced AE Signals
======================================================
Multiple zoom levels to see individual pulses clearly
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filepath):
    """Load a single .mat file."""
    mat_data = sio.loadmat(filepath)
    # Find the data key
    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    return mat_data[data_key].flatten(), data_key

def find_pulse_region(signal, threshold_percentile=99):
    """Find a region with clear pulse activity."""
    threshold = np.percentile(np.abs(signal), threshold_percentile)
    pulse_indices = np.where(np.abs(signal) > threshold)[0]
    if len(pulse_indices) > 0:
        # Return the first pulse location
        return pulse_indices[0]
    return 0

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load the 1.8mm data (clearer pulse patterns)
    filepath = os.path.join(base_path, 'data1.8mm.mat')
    signal, data_key = load_data(filepath)
    
    # Sampling rate (estimated ~1 MHz)
    fs = 1_000_000  # 1 MHz
    
    print(f"Loaded: {data_key}")
    print(f"Total samples: {len(signal):,}")
    print(f"Duration: {len(signal)/fs:.2f} seconds")
    
    # Find a region with pulse activity
    pulse_idx = find_pulse_region(signal)
    print(f"Found pulse activity around sample: {pulse_idx:,}")
    
    # ==========================================
    # FIGURE 1: Overview at different zoom levels
    # ==========================================
    fig1, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1 second window (1,000,000 samples)
    start = max(0, pulse_idx - 500000)
    window_1sec = signal[start:start + 1_000_000]
    time_1sec = np.arange(len(window_1sec)) / fs * 1000  # ms
    axes[0].plot(time_1sec, window_1sec, 'b-', linewidth=0.3)
    axes[0].set_title(f'1-Second Window (1,000,000 samples) - data1.8mm', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 100 ms window (100,000 samples)
    window_100ms = signal[pulse_idx:pulse_idx + 100_000]
    time_100ms = np.arange(len(window_100ms)) / fs * 1000  # ms
    axes[1].plot(time_100ms, window_100ms, 'b-', linewidth=0.5)
    axes[1].set_title('100ms Window (100,000 samples) - Zoom Level 1', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 10 ms window (10,000 samples)
    window_10ms = signal[pulse_idx:pulse_idx + 10_000]
    time_10ms = np.arange(len(window_10ms)) / fs * 1000000  # us
    axes[2].plot(time_10ms, window_10ms, 'b-', linewidth=0.5)
    axes[2].set_title('10ms Window (10,000 samples) - Zoom Level 2', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (us)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Q.Lin PD-Induced AE Signals - Progressive Zoom', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'zoom_overview.png'), dpi=150, bbox_inches='tight')
    print("Saved: zoom_overview.png")
    
    # ==========================================
    # FIGURE 2: 2001 samples window (as requested)
    # ==========================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    
    # Find multiple pulse regions for variety
    for i, ax in enumerate(axes2):
        # Use different starting points
        if i == 0:
            start_idx = pulse_idx
        else:
            # Scan forward to find more pulses
            search_start = pulse_idx + i * 5_000_000
            search_region = signal[search_start:search_start + 1_000_000]
            local_pulse = find_pulse_region(search_region) + search_start
            start_idx = local_pulse
        
        # Extract 2001 samples centered on pulse
        half_window = 1000
        center = start_idx + 500
        window_2001 = signal[center - half_window:center + half_window + 1]
        time_2001 = np.arange(len(window_2001)) / fs * 1000000  # microseconds
        
        ax.plot(time_2001, window_2001, 'b-', linewidth=1, marker='o', markersize=1, alpha=0.7)
        ax.set_title(f'2001 Samples Window #{i+1} (at sample {center:,})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (microseconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Q.Lin PD Signals - 2001 Sample Windows (2.001 ms at 1 MHz)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'window_2001_samples.png'), dpi=150, bbox_inches='tight')
    print("Saved: window_2001_samples.png")
    
    # ==========================================
    # FIGURE 3: Individual Pulse Detail
    # ==========================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    axes3 = axes3.flatten()
    
    # Find the largest pulse and zoom in
    # Scan for the top 6 largest pulses
    abs_signal = np.abs(signal[:50_000_000])  # First 50 million samples
    pulse_threshold = np.percentile(abs_signal, 99.99)
    large_pulse_indices = np.where(abs_signal > pulse_threshold)[0]
    
    # Get unique pulse regions (at least 1000 samples apart)
    unique_pulses = [large_pulse_indices[0]] if len(large_pulse_indices) > 0 else [0]
    for idx in large_pulse_indices[1:]:
        if len(unique_pulses) >= 6:
            break
        if all(abs(idx - p) > 1000 for p in unique_pulses):
            unique_pulses.append(idx)
    
    # Fill in if we don't have enough
    while len(unique_pulses) < 6:
        unique_pulses.append(unique_pulses[-1] + 10000)
    
    for i, ax in enumerate(axes3):
        center = unique_pulses[i]
        
        # 100-sample window (0.1 ms = 100 us)
        half = 50
        start = max(0, center - half)
        end = min(len(signal), center + half)
        window = signal[start:end]
        time_us = (np.arange(len(window)) + start) / fs * 1000000
        
        ax.plot(time_us, window, 'b-', linewidth=1.5, marker='o', markersize=3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Individual Pulse #{i+1}\n(100 samples, center @ {center:,})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (microseconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Mark the peak
        peak_idx = np.argmax(np.abs(window))
        ax.scatter([time_us[peak_idx]], [window[peak_idx]], color='red', s=100, zorder=5, label='Peak')
    
    plt.suptitle('Q.Lin PD Signals - Individual Pulse Detail (100 samples each)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'individual_pulses.png'), dpi=150, bbox_inches='tight')
    print("Saved: individual_pulses.png")
    
    # ==========================================
    # FIGURE 4: Comparison across all fault types
    # ==========================================
    files = {
        '1.0mm': 'data1.0mm.mat',
        '1.8mm': 'data1.8mm.mat',
        '2.0mm': 'data2.0mm.mat',
        '2.5mm': 'data2.5mm.mat'
    }
    
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    axes4 = axes4.flatten()
    
    for idx, (label, filename) in enumerate(files.items()):
        filepath_i = os.path.join(base_path, filename)
        if os.path.exists(filepath_i):
            sig, _ = load_data(filepath_i)
            
            # Find a pulse region
            pulse_loc = find_pulse_region(sig)
            
            # Show 500 samples around the pulse
            half = 250
            center = pulse_loc + 500
            window = sig[center - half:center + half]
            time_axis = np.arange(len(window)) / fs * 1000000
            
            axes4[idx].plot(time_axis, window, 'b-', linewidth=1, marker='o', markersize=2, alpha=0.7)
            axes4[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes4[idx].set_title(f'{label} - 500 Samples (0.5 ms)', fontsize=11, fontweight='bold')
            axes4[idx].set_xlabel('Time (microseconds)')
            axes4[idx].set_ylabel('Amplitude')
            axes4[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Q.Lin PD Signals - Pulse Comparison Across Fault Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'pulse_comparison_all_types.png'), dpi=150, bbox_inches='tight')
    print("Saved: pulse_comparison_all_types.png")
    
    print("\n[DONE] All visualizations saved!")

if __name__ == "__main__":
    main()
