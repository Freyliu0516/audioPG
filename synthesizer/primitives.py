import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import random
from scipy import signal


def harmonic_additive_synthesis(f0, duration, sr=16000, gamma=2.0, max_harmonics=50):
    """
    Implements harmonic additive synthesis with k^(-gamma) decay.
    Formula (3) from the paper: A_k = k^(-gamma)
    
    Args:
        f0: Fundamental frequency in Hz
        duration: Duration in seconds
        sr: Sampling rate
        gamma: Decay exponent
        max_harmonics: Maximum number of harmonics to include
    
    Returns:
        audio signal as numpy array
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal_out = np.zeros_like(t)
    
    # Calculate harmonics up to max_harmonics or until frequency exceeds Nyquist
    max_freq = sr // 2
    num_harmonics = min(max_harmonics, int(max_freq // f0))
    
    for k in range(1, num_harmonics + 1):
        # Amplitude follows k^(-gamma) decay
        amp = k ** (-gamma)
        freq = k * f0
        
        # Skip harmonics that exceed Nyquist frequency
        if freq >= max_freq:
            break
            
        harmonic = amp * np.sin(2 * np.pi * freq * t)
        signal_out += harmonic
    
    # Normalize to prevent clipping
    if np.max(np.abs(signal_out)) > 0:
        signal_out = signal_out / np.max(np.abs(signal_out))
    
    return signal_out


def fm_synthesis(carrier_freq, mod_freq_ratio, modulation_index, duration, sr=16000):
    """
    Implements Frequency Modulation synthesis.
    Formula (4) from the paper: x(t) = A sin(2πfc*t + β*sin(2πfm*t))
    
    Args:
        carrier_freq: Carrier frequency in Hz
        mod_freq_ratio: Ratio of modulator to carrier frequency (fm/fc)
        modulation_index: FM index (β)
        duration: Duration in seconds
        sr: Sampling rate
    
    Returns:
        audio signal as numpy array
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Calculate modulator frequency
    mod_freq = carrier_freq * mod_freq_ratio
    
    # FM synthesis formula: x(t) = A sin(2πfc*t + β*sin(2πfm*t))
    carrier_phase = 2 * np.pi * carrier_freq * t
    modulator_phase = modulation_index * np.sin(2 * np.pi * mod_freq * t)
    
    signal_out = np.sin(carrier_phase + modulator_phase)
    
    return signal_out


def broadband_pulse_wave(waveform_type='sawtooth', frequency=440, duration=1.0, sr=16000):
    """
    Generates broadband pulse waves (sawtooth, square, triangle).
    
    Args:
        waveform_type: Type of geometric waveform ('sawtooth', 'square', 'triangle')
        frequency: Fundamental frequency in Hz
        duration: Duration in seconds
        sr: Sampling rate
    
    Returns:
        audio signal as numpy array
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    period = 1.0 / frequency
    
    # Generate different waveforms
    if waveform_type == 'sawtooth':
        # Sawtooth wave: linear rise and fall
        signal_out = 2 * (t * frequency - np.floor(0.5 + t * frequency))
    elif waveform_type == 'square':
        # Square wave: alternates between high and low
        signal_out = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform_type == 'triangle':
        # Triangle wave: linear rise and fall in triangular pattern
        signal_out = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    else:
        raise ValueError(f"Unsupported waveform type: {waveform_type}")
    
    return signal_out


def generate_noise(duration, sr=16000, noise_type='white'):
    """
    Generate different types of noise.
    
    Args:
        duration: Duration in seconds
        sr: Sampling rate
        noise_type: Type of noise ('white', 'pink', 'brown')
    
    Returns:
        noise signal as numpy array
    """
    n_samples = int(sr * duration)
    
    if noise_type == 'white':
        # White noise: flat power spectrum
        noise = np.random.normal(0, 1, n_samples)
    elif noise_type == 'pink':
        # Pink noise: 1/f power spectrum
        noise = generate_pink_noise(n_samples)
    elif noise_type == 'brown':
        # Brown noise: 1/f^2 power spectrum
        white_noise = np.random.normal(0, 1, n_samples)
        noise = np.cumsum(white_noise)
        # Normalize to prevent overflow
        noise = noise / np.max(np.abs(noise)) if np.max(np.abs(noise)) > 0 else noise
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    return noise


def generate_pink_noise(n_samples, n_filters=20):
    """
    Generate pink noise using the Voss-McCartney algorithm.
    
    Args:
        n_samples: Number of samples to generate
        n_filters: Number of filters for approximation
    
    Returns:
        pink noise signal as numpy array
    """
    # Initialize output
    noise = np.zeros(n_samples)
    
    # Generate multiple random sequences with different update rates
    for i in range(n_filters):
        # Each sequence updates at a different rate
        update_every = 2 ** i
        n_updates = int(np.ceil(n_samples / update_every))
        
        # Generate random values for this sequence
        seq = np.random.normal(0, 1, n_updates)
        
        # Repeat values to fill the full length
        repeated_seq = np.repeat(seq, update_every)[:n_samples]
        
        # Add to the total
        noise += repeated_seq
    
    # Normalize
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    
    return noise


def apply_adsr_envelope(signal, attack_time, decay_time, sustain_level, release_time, sr=16000):
    """
    Apply ADSR envelope to an audio signal.
    
    Args:
        signal: Input audio signal
        attack_time: Attack time in seconds
        decay_time: Decay time in seconds
        sustain_level: Sustain level (0-1)
        release_time: Release time in seconds
        sr: Sampling rate
    
    Returns:
        Enveloped signal
    """
    n_samples = len(signal)
    total_duration = n_samples / sr
    
    # Calculate sample points for each phase
    attack_samples = int(attack_time * sr)
    decay_samples = int(decay_time * sr)
    release_samples = int(release_time * sr)
    
    # Ensure we don't exceed the signal length
    attack_samples = min(attack_samples, n_samples)
    decay_samples = min(decay_samples, n_samples - attack_samples)
    release_start = min(n_samples - release_samples, n_samples)
    
    envelope = np.ones(n_samples)
    
    # Attack phase: 0 to 1
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay phase: 1 to sustain_level
    decay_end = min(attack_samples + decay_samples, n_samples)
    if decay_samples > 0 and decay_end > attack_samples:
        envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, 
                                                        decay_end - attack_samples)
    
    # Sustain phase: constant at sustain_level
    sustain_start = attack_samples + decay_samples
    release_start_idx = max(sustain_start, n_samples - release_samples)
    if release_start_idx > sustain_start and release_start_idx < n_samples:
        envelope[sustain_start:release_start_idx] = sustain_level
    
    # Release phase: from sustain_level to 0
    if release_samples > 0 and release_start_idx < n_samples:
        envelope[release_start_idx:] = np.linspace(sustain_level, 0, 
                                                  n_samples - release_start_idx)
    
    return signal * envelope


def create_dynamic_audio_segment(fundamental_freq, duration=10.24, sr=16000):
    """
    Create a dynamic audio segment with multiple events and temporal variations.
    This implementation closely follows the original 2.3 advanced_physics_synth function.
    
    Args:
        fundamental_freq: Base frequency for the audio
        duration: Duration in seconds
        sr: Sampling rate
    
    Returns:
        Dynamic audio signal with multiple events and temporal variations
    """
    t = np.arange(duration * sr) / sr
    num_samples = len(t)
    
    # Randomly choose synthesis mode (matching original implementation)
    synth_mode = random.choice(['harmonic', 'fm', 'pulse'])
    
    if synth_mode == 'harmonic':
        # Harmonic additive synthesis (matching original implementation)
        harmonics_count = random.randint(3, 15)
        waveform = np.zeros_like(t)
        for i in range(harmonics_count):
            # Harmonic decay + random phase
            strength = (1.0 / (i + 1)**random.uniform(0.5, 1.5)) * random.uniform(0.5, 1.2)
            phase = random.uniform(0, 2*np.pi)
            # Frequency tuning (simulate imperfect harmonics)
            freq = fundamental_freq * (i + 1) * random.uniform(0.995, 1.005)
            waveform += strength * np.sin(2 * np.pi * freq * t + phase)
            
    elif synth_mode == 'fm':
        # FM synthesis (matching original implementation)
        modulation_index = random.uniform(0.5, 8.0)
        mod_ratio = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.14])  # Simple ratios
        mod_freq = fundamental_freq * mod_ratio
        modulator = modulation_index * np.sin(2 * np.pi * mod_freq * t)
        waveform = np.sin(2 * np.pi * fundamental_freq * t + modulator)
        
    else:  # pulse
        # Pulse/Saw simulation (matching original implementation)
        if random.random() > 0.5:
            # Simple sawtooth approximation
            waveform = 2.0 * (t * fundamental_freq - np.floor(t * fundamental_freq + 0.5))
        else:
            # Simple square wave
            waveform = np.sign(np.sin(2 * np.pi * fundamental_freq * t))

    # --- 1. Transient (Burst) ---
    if random.random() > 0.4:
        burst_len = int(random.uniform(0.01, 0.1) * sr)
        if burst_len < num_samples:
            start_idx = random.randint(0, num_samples - burst_len)
            burst = (np.random.rand(burst_len) * 2 - 1)
            waveform[start_idx:start_idx+burst_len] += burst * random.uniform(0.5, 2.0)

    # --- 2. ADSR Envelope (Key! Makes sound dynamic) ---
    # Randomly generate multiple events instead of full duration
    # AudioMAE needs to learn temporal changes, can't be all steady-state
    num_events = random.randint(1, 5)
    final_waveform = np.zeros_like(t)
    
    for _ in range(num_events):
        event_len_sec = random.uniform(0.5, duration/num_events)
        event_len = int(event_len_sec * sr)
        if event_len > num_samples: event_len = num_samples
        
        # Simple ADSR
        attack = int(event_len * random.uniform(0.01, 0.3))
        decay = int(event_len * random.uniform(0.1, 0.4))
        sustain_len = max(0, event_len - attack - decay)
        
        env = np.ones(event_len)
        env[:attack] = np.linspace(0, 1, attack)
        env[attack:attack+decay] = np.linspace(1, 0.6, decay)
        env[attack+decay:] = np.linspace(0.6, 0, sustain_len)
        
        # Extract a segment of the base waveform
        start_pos = random.randint(0, num_samples - event_len)
        segment = waveform[start_pos:start_pos+event_len] * env
        
        # Place at random position on timeline
        place_pos = random.randint(0, num_samples - event_len)
        final_waveform[place_pos:place_pos+event_len] += segment

    waveform = final_waveform

    return waveform