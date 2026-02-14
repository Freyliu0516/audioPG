import numpy as np
import torch
import torchaudio
import random
from synthesizer.primitives import (
    harmonic_additive_synthesis, fm_synthesis, broadband_pulse_wave, 
    generate_noise, apply_adsr_envelope
)
from synthesizer.dynamics import (
    inject_transient_noise, 
    place_multiple_events
)
from synthesizer.primitives import create_dynamic_audio_segment
from synthesizer.modifiers import apply_full_postprocessing_chain


class WorldModelSynthesizer:
    """
    The complete World Model synthesizer that implements the procedural audio generation
    as described in the AudioPG paper. This class integrates all components:
    - Physical synthesis primitives (harmonic, FM, broadband)
    - Temporal dynamics (ADSR, event placement, transients)
    - Post-processing (filtering, noise, normalization)
    """
    
    def __init__(self, sr=16000, duration=10.24):
        """
        Initialize the World Model Synthesizer.
        
        Args:
            sr: Sampling rate (default 16000 Hz as per paper)
            duration: Duration of generated audio in seconds (default 10.24s as per paper)
        """
        self.sr = sr
        self.duration = duration
        self.n_samples = int(sr * duration)
    
    def generate_single_event(self, fundamental_freq=None):
        """
        Generate a single audio event using random synthesis method.
        
        Args:
            fundamental_freq: Base frequency (if None, chosen randomly)
        
        Returns:
            Generated audio signal as numpy array
        """
        if fundamental_freq is None:
            # Random fundamental frequency as per 2.3 implementation
            fundamental_freq = random.uniform(50.0, 2000.0)
        
        # Randomly select synthesis method
        synth_method = random.choice(['harmonic', 'fm', 'pulse', 'noise'])
        
        if synth_method == 'harmonic':
            # Harmonic additive synthesis with random gamma
            gamma = random.uniform(0.5, 2.0)
            signal = harmonic_additive_synthesis(
                f0=fundamental_freq,
                duration=self.duration,
                sr=self.sr,
                gamma=gamma
            )
        elif synth_method == 'fm':
            # FM synthesis with random parameters
            mod_ratio = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.14])
            modulation_index = random.uniform(0.5, 8.0)
            signal = fm_synthesis(
                carrier_freq=fundamental_freq,
                mod_freq_ratio=mod_ratio,
                modulation_index=modulation_index,
                duration=self.duration,
                sr=self.sr
            )
        elif synth_method == 'pulse':
            # Broadband pulse wave
            waveform_type = random.choice(['sawtooth', 'square', 'triangle'])
            signal = broadband_pulse_wave(
                waveform_type=waveform_type,
                frequency=fundamental_freq,
                duration=self.duration,
                sr=self.sr
            )
        else:  # noise
            # Pure noise with some shaping
            signal = generate_noise(self.duration, self.sr, noise_type='pink')
        
        return signal
    
    def generate_with_temporal_dynamics(self, fundamental_freq=None):
        """
        Generate audio with temporal dynamics (multiple events, ADSR, etc.).
        
        Args:
            fundamental_freq: Base frequency (if None, chosen randomly)
        
        Returns:
            Generated audio signal with temporal dynamics
        """
        if fundamental_freq is None:
            fundamental_freq = random.uniform(50.0, 2000.0)
        
        # Create dynamic segment with multiple events
        signal = create_dynamic_audio_segment(
            fundamental_freq=fundamental_freq,
            duration=self.duration,
            sr=self.sr
        )
        
        return signal
    
    def generate_complete_sample(self, fundamental_freq=None):
        """
        Generate a complete audio sample with all processing stages applied.
        This is the main method that implements the full world model.
        
        Args:
            fundamental_freq: Base frequency (if None, chosen randomly)
        
        Returns:
            Complete processed audio signal
        """
        # Start with a dynamically generated segment
        signal = self.generate_with_temporal_dynamics(fundamental_freq)
        
        # Apply full post-processing chain
        processed_signal = apply_full_postprocessing_chain(signal, self.sr)
        
        # Convert to numpy if it's a tensor for max calculation
        if torch.is_tensor(processed_signal):
            abs_max = torch.max(torch.abs(processed_signal)).item()
            if abs_max > 0:
                # Normalize to prevent clipping while preserving dynamics
                if abs_max > 1.0:
                    processed_signal = processed_signal / abs_max
        else:
            if np.max(np.abs(processed_signal)) > 0:
                # Normalize to prevent clipping while preserving dynamics
                max_val = np.max(np.abs(processed_signal))
                if max_val > 1.0:
                    processed_signal = processed_signal / max_val
        
        return processed_signal
    
    def convert_to_melspec(self, audio_signal):
        """
        Convert audio signal to mel-spectrogram as used in the AudioMAE pipeline.
        
        Args:
            audio_signal: Generated audio signal
        
        Returns:
            Mel-spectrogram tensor ready for model input
        """
        # Convert numpy array to torch tensor
        if isinstance(audio_signal, np.ndarray):
            waveform = torch.from_numpy(audio_signal).float()
        else:
            waveform = audio_signal.float()
        
        # Ensure proper shape (1, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Generate mel-spectrogram using Kaldi fbank (as in 2.3 implementation)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, 
            htk_compat=True, 
            sample_frequency=self.sr, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=128, 
            dither=0.0, 
            frame_shift=10
        )
        
        # Pad or trim to target length (1024 as in 2.3 implementation)
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:target_length, :]
        
        # Normalize as in 2.3 implementation
        fbank = (fbank - (-4.26)) / (4.56 * 2)
        
        # Add channel dimension
        fbank = fbank.unsqueeze(0)
        
        return fbank
    
    def generate_batch(self, batch_size, return_melspec=True):
        """
        Generate a batch of audio samples.
        
        Args:
            batch_size: Number of samples to generate
            return_melspec: Whether to return mel-spectrograms or raw audio
        
        Returns:
            Batch of generated samples (either mel-spectrograms or raw audio)
        """
        batch = []
        
        for _ in range(batch_size):
            # Generate complete sample
            audio = self.generate_complete_sample()
            
            if return_melspec:
                # Convert to mel-spectrogram
                melspec = self.convert_to_melspec(audio)
                batch.append(melspec)
            else:
                batch.append(audio)
        
        if return_melspec:
            # Stack mel-spectrograms into batch tensor
            batch_tensor = torch.stack(batch)
        else:
            # Convert to tensor and stack audio signals
            batch_tensor = torch.stack([torch.from_numpy(a).float() if isinstance(a, np.ndarray) 
                                        else a.float() for a in batch])
        
        return batch_tensor


class PhysicsBasedDataset(torch.utils.data.Dataset):
    """
    Dataset class that generates samples on-the-fly using the World Model Synthesizer,
    as required by the paper's "Generated On-the-fly" approach.
    """
    
    def __init__(self, epoch_len=5000, sr=16000, duration=10.24):
        """
        Initialize the physics-based dataset.
        
        Args:
            epoch_len: Number of samples per epoch
            sr: Sampling rate
            duration: Duration of each sample in seconds
        """
        self.synthesizer = WorldModelSynthesizer(sr=sr, duration=duration)
        self.epoch_len = epoch_len
    
    def __len__(self):
        return self.epoch_len
    
    def __getitem__(self, idx):
        """
        Generate a sample on-the-fly.
        
        Args:
            idx: Sample index (not used since generation is random)
        
        Returns:
            Tuple of (mel-spectrogram, dummy_label)
        """
        # Generate complete sample with all processing
        audio = self.synthesizer.generate_complete_sample()
        
        # Convert to mel-spectrogram
        melspec = self.synthesizer.convert_to_melspec(audio)
        
        # Return with dummy label (0) as in 2.3 implementation
        return melspec, torch.tensor(0)


def demo_world_model():
    """
    Demonstration function to showcase the World Model Synthesizer.
    """
    print("🧪 Demonstrating World Model Synthesizer...")
    
    # Create synthesizer
    synthesizer = WorldModelSynthesizer()
    
    # Generate a few samples
    for i in range(3):
        print(f"Generating sample {i+1}/3...")
        sample = synthesizer.generate_complete_sample()
        melspec = synthesizer.convert_to_melspec(sample)
        
        print(f"  Sample shape: {sample.shape}")
        print(f"  Mel-spec shape: {melspec.shape}")
        
        # Handle both numpy arrays and torch tensors
        if torch.is_tensor(sample):
            min_val = torch.min(sample).item()
            max_val = torch.max(sample).item()
        else:
            min_val = np.min(sample)
            max_val = np.max(sample)
        print(f"  Sample amplitude range: [{min_val:.3f}, {max_val:.3f}]")
    
    print("✅ World Model demonstration completed!")


if __name__ == "__main__":
    demo_world_model()