import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    """ESC-50 dataset loader."""
    
    def __init__(self, root_dir, fold=1, train=True, target_length=1024, sr=16000):
        """
        Args:
            root_dir: Root directory of ESC-50 dataset
            fold: Which fold to use (1-5)
            train: Whether to use training or validation split
            target_length: Target length for spectrograms (time frames)
            sr: Sampling rate
        """
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.meta_path = os.path.join(root_dir, 'meta', 'esc50.csv')
        self.target_length = target_length
        self.sr = sr
        self.train = train
        
        # Load metadata
        df = pd.read_csv(self.meta_path)
        
        # Split based on fold
        if train:
            self.df = df[df['fold'] != fold]
        else:
            self.df = df[df['fold'] == fold]
            
        # Create category to index mapping
        self.categories = sorted(df['category'].unique())
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (spectrogram, label) where label is the class index
        """
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['filename'])
        label = self.cat2idx[row['category']]
        
        # Load audio
        waveform, orig_sr = torchaudio.load(wav_path)
        
        # Resample if needed
        if orig_sr != self.sr:
            resampler = T.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or crop to 10.24 seconds (required for 1024 time frames at 100Hz frame rate)
        target_samples = int(self.sr * 10.24)  # 163840 samples at 16kHz
        current_samples = waveform.shape[1]
        
        if current_samples < target_samples:
            # Pad with zeros
            pad_length = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif current_samples > target_samples:
            # Crop to target length
            waveform = waveform[:, :target_samples]
        
        # Convert to mel-spectrogram
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
        
        # Pad or trim to target length (1024 time frames)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:self.target_length, :]
        
        # Normalize as done in the original implementation
        fbank = (fbank - (-4.26)) / (4.56 * 2)
        
        # Add channel dimension
        fbank = fbank.unsqueeze(0)
        
        return fbank, label


def get_esc50_stats():
    """Print statistics about ESC-50 dataset."""
    print("📊 ESC-50 Dataset Statistics:")
    print("- 2,000 labeled sound excerpts")
    print("- 50 classes with 40 examples each")
    print("- 5-fold cross validation setup")
    print("- 5-second audio clips")
    print("- Classes: e.g., dog bark, rain, sea waves, baby cry, clock tick, etc.")


if __name__ == "__main__":
    # Example usage
    get_esc50_stats()