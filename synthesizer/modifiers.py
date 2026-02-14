import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import random


def apply_lowpass_filter(signal, sr=16000, filter_prob=0.7):
    """
    Apply low-pass filtering with random cutoff frequency as in 2.3 implementation.
    
    Args:
        signal: Input signal (numpy array or torch tensor)
        sr: Sampling rate
        filter_prob: Probability of applying the filter
    
    Returns:
        Filtered signal
    """
    if random.random() > filter_prob:
        # Convert to tensor if it's numpy
        if isinstance(signal, np.ndarray):
            signal_tensor = torch.from_numpy(signal).float()
        else:
            signal_tensor = signal.float()
        
        # Reshape for torchaudio biquad filter (batch, samples)
        if signal_tensor.dim() == 1:
            signal_tensor = signal_tensor.unsqueeze(0)
        
        # Random cutoff frequency between 200Hz and 6000Hz
        cutoff = random.uniform(200, 6000)
        signal_filtered = F.lowpass_biquad(signal_tensor, sr, cutoff_freq=cutoff)
        
        # Return to original shape/dtype
        if signal_filtered.shape[0] == 1:
            signal_filtered = signal_filtered.squeeze(0)
        
        if isinstance(signal, np.ndarray):
            return signal_filtered.numpy()
        else:
            return signal_filtered
    
    # No filter applied, return original
    return signal


def add_background_noise(signal, sr=16000, noise_prob=0.8):
    """
    Add background noise as in 2.3 implementation.
    
    Args:
        signal: Input signal
        sr: Sampling rate
        noise_prob: Probability of adding background noise
    
    Returns:
        Signal with possible background noise
    """
    signal_len = len(signal) if isinstance(signal, np.ndarray) else signal.shape[-1]
    
    if random.random() > noise_prob:
        # Generate background noise
        noise_level = random.uniform(0.001, 0.02)
        
        if isinstance(signal, np.ndarray):
            noise = (np.random.rand(signal_len) * 2 - 1) * noise_level
            signal_with_noise = signal + noise
        else:
            # Handle torch tensors
            noise = (torch.rand_like(signal) * 2 - 1) * noise_level
            signal_with_noise = signal + noise
    else:
        signal_with_noise = signal
    
    return signal_with_noise


def peak_normalize(signal, normalize_prob=0.95):
    """
    Apply peak normalization (auto-gain) as mentioned in the paper.
    
    Args:
        signal: Input signal
        normalize_prob: Probability of applying normalization
    
    Returns:
        Peak normalized signal
    """
    if random.random() > normalize_prob:
        # Check if signal is numpy array or torch tensor
        if isinstance(signal, np.ndarray):
            max_abs = np.max(np.abs(signal))
            if max_abs > 0:
                signal_normalized = signal / max_abs
            else:
                signal_normalized = signal
        else:
            max_abs = torch.max(torch.abs(signal))
            if max_abs > 0:
                signal_normalized = signal / max_abs
            else:
                signal_normalized = signal
    else:
        signal_normalized = signal
    
    return signal_normalized


def apply_spectral_damping(signal, sr=16000):
    """
    Apply spectral damping with random cutoff as described in the paper.
    This is essentially the low-pass filter with random cutoff frequency fc ~ U(200, 6000).
    
    Args:
        signal: Input signal
        sr: Sampling rate
    
    Returns:
        Spectrally damped signal
    """
    # Random cutoff frequency uniformly distributed between 200 and 6000 Hz
    cutoff_freq = random.uniform(200, 6000)
    
    # Convert to tensor if it's numpy
    if isinstance(signal, np.ndarray):
        signal_tensor = torch.from_numpy(signal).float()
    else:
        signal_tensor = signal.float()
    
    # Reshape for torchaudio biquad filter (batch, samples)
    if signal_tensor.dim() == 1:
        signal_tensor = signal_tensor.unsqueeze(0)
    
    # Apply lowpass biquad filter
    signal_damped = F.lowpass_biquad(signal_tensor, sr, cutoff_freq=cutoff_freq)
    
    # Return to original shape/dtype
    if signal_damped.shape[0] == 1:
        signal_damped = signal_damped.squeeze(0)
    
    if isinstance(signal, np.ndarray):
        return signal_damped.numpy()
    else:
        return signal_damped


def apply_random_gain(signal):
    """
    Apply random gain variation similar to the 2.3 implementation.
    
    Args:
        signal: Input signal
    
    Returns:
        Signal with random gain applied
    """
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        
        if isinstance(signal, np.ndarray):
            return signal * gain
        else:
            return signal * gain
    else:
        return signal


def mix_with_noise(signal, lambda_n=None, sr=16000):
    """
    Mix signal with scaled noise η(t) as in the formula λ_n * η(t).
    
    Args:
        signal: Input signal
        lambda_n: Scaling factor for noise (if None, chosen randomly)
        sr: Sampling rate
    
    Returns:
        Mixed signal with added scaled noise
    """
    if lambda_n is None:
        # Random scaling factor based on the range in the 2.3 code
        lambda_n = random.uniform(0.001, 0.02)
    
    signal_len = len(signal) if isinstance(signal, np.ndarray) else signal.shape[-1]
    
    if isinstance(signal, np.ndarray):
        noise = np.random.randn(signal_len)
        mixed_signal = signal + lambda_n * noise
    else:
        noise = torch.randn_like(signal)
        mixed_signal = signal + lambda_n * noise
    
    return mixed_signal


def apply_full_postprocessing_chain(signal, sr=16000):
    """
    Apply the complete post-processing chain as described in the paper and original implementation:
    1. Spectral damping (low-pass filter)
    2. Background noise addition
    3. Peak normalization (auto-gain)
    
    Args:
        signal: Input signal
        sr: Sampling rate
    
    Returns:
        Fully processed signal
    """
    # Convert to tensor for processing (as in original implementation)
    if isinstance(signal, np.ndarray):
        signal_tensor = torch.from_numpy(signal).float()
    else:
        signal_tensor = signal.float()
    
    # Ensure proper shape (1, time)
    if signal_tensor.dim() == 1:
        signal_tensor = signal_tensor.unsqueeze(0)
    
    # Step 1: Apply spectral damping (low-pass filter) - as in original implementation
    if random.random() > 0.3:  # Apply filter with 70% probability
        cutoff = random.uniform(200, 6000)
        signal_tensor = F.lowpass_biquad(signal_tensor, sr, cutoff_freq=cutoff)

    # Step 2: Add background noise - as in original implementation
    signal_len = signal_tensor.shape[-1]
    noise = (torch.rand(1, signal_len) * 2 - 1) * random.uniform(0.001, 0.02)
    signal_tensor += noise

    # Step 3: Apply peak normalization (auto-gain)
    if signal_tensor.abs().max() > 0:
        signal_tensor = signal_tensor / signal_tensor.abs().max()
    
    # Return as squeezed tensor (remove batch dimension)
    return signal_tensor.squeeze(0)