import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset


class UrbanSound8KDataset(Dataset):
    """UrbanSound8K dataset loader."""
    
    def __init__(self, root_dir, fold=1, train=True, target_length=1024, sr=16000):
        """
        Args:
            root_dir: Root directory of UrbanSound8K dataset
            fold: Which fold to use (1-10)
            train: Whether to use training or validation split
            target_length: Target length for spectrograms (time frames)
            sr: Sampling rate
        """
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, 'metadata', 'UrbanSound8K.csv')
        self.target_length = target_length
        self.sr = sr
        self.train = train
        
        # Load metadata
        df = pd.read_csv(self.meta_path)
        
        # Split based on fold (standard: folds 1-9 for training, fold 10 for testing)
        if train:
            self.df = df[df['fold'] != fold]
        else:
            self.df = df[df['fold'] == fold]
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (spectrogram, label) where label is the class index
        """
        row = self.df.iloc[idx]
        
        # Construct file path: audio/fold{fold_number}/{file_name}
        fold_str = f"fold{row['fold']}"
        file_path = os.path.join(self.root_dir, 'audio', fold_str, row['slice_file_name'])
        label = row['classID']  # Class ID is already numeric (0-9)
        
        try:
            # Load audio
            waveform, orig_sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo (common in UrbanSound8K)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if orig_sr != self.sr:
                resampler = T.Resample(orig_sr, self.sr)
                waveform = resampler(waveform)
            
            # Pad or crop to 10.24 seconds (required for 1024 time frames at 100Hz frame rate)
            target_samples = int(self.sr * 10.24)  # 163840 samples at 16kHz
            current_samples = waveform.shape[1]
            
            if current_samples < target_samples:
                # Pad with zeros
                pad_length = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            elif current_samples > target_samples:
                # Crop to target length (take center crop)
                start_idx = (current_samples - target_samples) // 2
                waveform = waveform[:, start_idx:start_idx + target_samples]
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zero tensor if there's an error
            waveform = torch.zeros(1, int(self.sr * 10.24))
        
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


def get_us8k_stats():
    """Print statistics about UrbanSound8K dataset."""
    print("📊 UrbanSound8K Dataset Statistics:")
    print("- 8,732 labeled sound excerpts")
    print("- 10 classes with varying number of examples")
    print("- 10-fold cross validation setup")
    print("- Up to 4-second audio clips")
    print("- Classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, etc.")


if __name__ == "__main__":
    # Example usage
    get_us8k_stats()