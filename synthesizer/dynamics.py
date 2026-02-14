import numpy as np
import torch
import random
from synthesizer.primitives import apply_adsr_envelope


def generate_random_adsr_params():
    """
    Generate random ADSR parameters following the patterns observed in the 2.3 code.
    
    Returns:
        Dictionary with attack, decay, sustain, release times and sustain level
    """
    # Based on the 2.3 implementation patterns
    attack_time = random.uniform(0.01, 0.3)  # 10ms to 300ms
    decay_time = random.uniform(0.1, 0.4)    # 100ms to 400ms
    sustain_level = random.uniform(0.3, 0.8)  # 30% to 80%
    release_time = random.uniform(0.1, 0.5)   # 100ms to 500ms
    
    return {
        'attack_time': attack_time,
        'decay_time': decay_time,
        'sustain_level': sustain_level,
        'release_time': release_time
    }


def place_multiple_events(signal, duration=10.24, sr=16000, num_events_range=(1, 5)):
    """
    Place multiple events within the signal duration following the 2.3 implementation.
    
    Args:
        signal: Input signal to place events in
        duration: Total duration in seconds
        sr: Sampling rate
        num_events_range: Range of number of events to place (min, max)
    
    Returns:
        Signal with multiple events placed at random positions
    """
    total_samples = len(signal)
    final_signal = np.zeros_like(signal)
    
    # Randomly determine number of events
    num_events = random.randint(num_events_range[0], num_events_range[1])
    
    for _ in range(num_events):
        # Determine event length
        event_len_sec = random.uniform(0.5, duration / num_events)
        event_len = min(int(event_len_sec * sr), total_samples)
        
        # Get random ADSR parameters
        adsr_params = generate_random_adsr_params()
        
        # Create envelope
        env = np.ones(event_len)
        attack_samples = int(event_len * min(adsr_params['attack_time'], 1.0))
        decay_samples = int(event_len * min(adsr_params['decay_time'], 1.0))
        sustain_len = max(0, event_len - attack_samples - decay_samples)
        
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0 and attack_samples < event_len:
            env[attack_samples:attack_samples+decay_samples] = np.linspace(
                1, adsr_params['sustain_level'], decay_samples
            )
        if sustain_len > 0 and attack_samples + decay_samples < event_len:
            env[attack_samples+decay_samples:] = np.linspace(
                adsr_params['sustain_level'], 0, sustain_len
            )
        
        # Extract a segment from the input signal
        start_pos = random.randint(0, max(0, total_samples - event_len))
        segment = signal[start_pos:start_pos+event_len] * env
        
        # Place the segment at a random position in the final signal
        place_pos = random.randint(0, max(0, total_samples - event_len))
        final_signal[place_pos:place_pos+event_len] += segment
    
    return final_signal


def inject_transient_noise(signal, sr=16000, transient_prob=0.6):
    """
    Inject short bursts of white noise to simulate transients as in 2.3 implementation.
    
    Args:
        signal: Input signal
        sr: Sampling rate
        transient_prob: Probability of adding a transient burst
    
    Returns:
        Signal with possible transient injection
    """
    if random.random() > transient_prob:
        # Generate burst parameters
        burst_len = int(random.uniform(0.01, 0.1) * sr)  # 10ms to 100ms
        signal_len = len(signal)
        
        if burst_len < signal_len:
            start_idx = random.randint(0, signal_len - burst_len)
            burst = (np.random.rand(burst_len) * 2 - 1)  # White noise between -1 and 1
            burst_strength = random.uniform(0.5, 2.0)  # Variable strength
            
            # Add the burst to the signal
            signal_with_transient = signal.copy()
            signal_with_transient[start_idx:start_idx+burst_len] += burst * burst_strength
            
            return signal_with_transient
    
    # No transient added, return original signal
    return signal


def apply_random_temporal_modulation(signal, sr=16000):
    """
    Apply temporal modulation similar to the rolling technique seen in the 2.3 code.
    
    Args:
        signal: Input signal
        sr: Sampling rate
    
    Returns:
        Temporally modulated signal
    """
    # Random time shifting/rolling (similar to the wave augmentation in 2.3)
    if random.random() < 0.5:
        shift = random.randint(0, len(signal))
        signal = np.roll(signal, shift)
    
    return signal


