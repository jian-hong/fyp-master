"""
MATLAB .mat file loader for real PD data.

Loads data from Q.lin dataset and extracts noise segments for TGAN training.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Try to import scipy.io for .mat file loading
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import mat73 for MATLAB v7.3 files
try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False

# Try to import h5py for HDF5-based .mat files
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_mat_data(
    file_path: Union[str, Path],
    variable_name: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load data from a MATLAB .mat file.
    
    Handles both v7.3 (HDF5) and older .mat formats.
    
    Args:
        file_path: Path to .mat file
        variable_name: Specific variable to load. If None, loads all.
    
    Returns:
        Dictionary of variable names to numpy arrays
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"MAT file not found: {file_path}")
    
    data = {}
    
    # Try scipy.io first (for older .mat formats)
    if HAS_SCIPY:
        try:
            mat_data = sio.loadmat(str(file_path))
            # Filter out metadata keys
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    if variable_name is None or key == variable_name:
                        data[key] = np.array(value)
            if data:
                return data
        except NotImplementedError:
            # v7.3 format, try other methods
            pass
        except Exception as e:
            warnings.warn(f"scipy.io failed: {e}")
    
    # Try mat73 for v7.3 format
    if HAS_MAT73:
        try:
            mat_data = mat73.loadmat(str(file_path))
            for key, value in mat_data.items():
                if variable_name is None or key == variable_name:
                    data[key] = np.array(value)
            if data:
                return data
        except Exception as e:
            warnings.warn(f"mat73 failed: {e}")
    
    # Try h5py for HDF5-based files
    if HAS_H5PY:
        try:
            with h5py.File(str(file_path), 'r') as f:
                for key in f.keys():
                    if variable_name is None or key == variable_name:
                        data[key] = np.array(f[key])
            if data:
                return data
        except Exception as e:
            warnings.warn(f"h5py failed: {e}")
    
    if not data:
        raise RuntimeError(
            f"Could not load {file_path}. "
            "Please install scipy, mat73, or h5py: pip install scipy mat73 h5py"
        )
    
    return data


def load_qlin_dataset(
    data_dir: Union[str, Path]
) -> Dict[str, np.ndarray]:
    """
    Load the Q.lin dataset from the specified directory.
    
    Expected files:
    - data1.0mm.mat
    - data1.8mm.mat
    - data2.0mm.mat
    - data2.5mm.mat
    
    Args:
        data_dir: Path to "Datset from Q.lin" directory
    
    Returns:
        Dictionary mapping distance labels to data arrays
    """
    data_dir = Path(data_dir)
    
    expected_files = {
        '1.0mm': 'data1.0mm.mat',
        '1.8mm': 'data1.8mm.mat',
        '2.0mm': 'data2.0mm.mat',
        '2.5mm': 'data2.5mm.mat',
    }
    
    dataset = {}
    
    for label, filename in expected_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                data = load_mat_data(file_path)
                # Find the data variable (typically named similar to file)
                var_name = None
                for key in data.keys():
                    if 'data' in key.lower() or 'mm' in key.lower():
                        var_name = key
                        break
                
                if var_name is None and len(data) == 1:
                    var_name = list(data.keys())[0]
                
                if var_name:
                    dataset[label] = data[var_name]
                    print(f"Loaded {filename}: shape {dataset[label].shape}")
            except Exception as e:
                warnings.warn(f"Failed to load {filename}: {e}")
        else:
            warnings.warn(f"File not found: {file_path}")
    
    return dataset


def extract_noise_segments(
    signal: np.ndarray,
    segment_length: int = 256,
    threshold_factor: float = 0.1,
    min_segments: int = 10
) -> np.ndarray:
    """
    Extract noise-only segments from PD signals for TGAN training.
    
    Identifies regions where signal amplitude is consistently low,
    indicating background noise without PD pulses.
    
    Args:
        signal: 1D or 2D signal array (samples x time or just time)
        segment_length: Length of each noise segment
        threshold_factor: Fraction of max amplitude to consider as noise
        min_segments: Minimum number of segments to extract
    
    Returns:
        Array of noise segments (num_segments x segment_length)
    """
    # Handle 2D input
    if signal.ndim == 2:
        # Flatten or process each row
        all_segments = []
        for row in signal:
            segments = extract_noise_segments(
                row, segment_length, threshold_factor, min_segments=1
            )
            all_segments.append(segments)
        return np.vstack(all_segments)
    
    signal = np.asarray(signal).flatten()
    signal_length = len(signal)
    
    # Calculate amplitude envelope using sliding window
    window_size = segment_length // 4
    envelope = np.zeros(signal_length)
    
    for i in range(signal_length):
        start = max(0, i - window_size // 2)
        end = min(signal_length, i + window_size // 2)
        envelope[i] = np.max(np.abs(signal[start:end]))
    
    # Threshold for noise detection
    threshold = threshold_factor * np.max(np.abs(signal))
    
    # Find noise regions
    noise_mask = envelope < threshold
    
    # Extract non-overlapping segments from noise regions
    segments = []
    i = 0
    
    while i < signal_length - segment_length:
        if np.all(noise_mask[i:i + segment_length]):
            segments.append(signal[i:i + segment_length])
            i += segment_length
        else:
            i += segment_length // 2
    
    # If not enough segments, lower threshold
    if len(segments) < min_segments:
        # Fall back to random sampling with amplitude check
        attempts = 0
        while len(segments) < min_segments and attempts < 1000:
            start = np.random.randint(0, signal_length - segment_length)
            segment = signal[start:start + segment_length]
            
            # Check if relatively low amplitude
            if np.max(np.abs(segment)) < 0.3 * np.max(np.abs(signal)):
                segments.append(segment)
            attempts += 1
    
    if len(segments) == 0:
        raise ValueError("Could not extract any noise segments from signal")
    
    return np.array(segments)


def prepare_tgan_training_data(
    data_dir: Union[str, Path],
    segment_length: int = 256,
    max_segments_per_file: int = 1000
) -> np.ndarray:
    """
    Prepare training data for TGAN from Q.lin dataset.
    
    Extracts noise segments from all available files.
    
    Args:
        data_dir: Path to data directory
        segment_length: Length of noise segments
        max_segments_per_file: Maximum segments to extract per file
    
    Returns:
        Array of noise segments for TGAN training
    """
    dataset = load_qlin_dataset(data_dir)
    
    all_segments = []
    
    for label, data in dataset.items():
        try:
            segments = extract_noise_segments(
                data,
                segment_length=segment_length,
                min_segments=10
            )
            
            # Limit segments per file
            if len(segments) > max_segments_per_file:
                indices = np.random.choice(
                    len(segments),
                    max_segments_per_file,
                    replace=False
                )
                segments = segments[indices]
            
            all_segments.append(segments)
            print(f"Extracted {len(segments)} noise segments from {label}")
            
        except Exception as e:
            warnings.warn(f"Failed to extract from {label}: {e}")
    
    if not all_segments:
        raise ValueError("Could not extract noise segments from any file")
    
    return np.vstack(all_segments)
