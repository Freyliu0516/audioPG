import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset
from collections import Counter


class SpeechCommandsV2Dataset(Dataset):
    """Google Speech Commands V2 dataset loader."""
    
    def __init__(self, root_dir, subset='train', target_length=1024, sr=16000):
        """
        Args:
            root_dir: Root directory of Speech Commands V2 dataset
            subset: 'train', 'validation', or 'testing'
            target_length: Target length for spectrograms (time frames)
            sr: Sampling rate
        """
        self.root_dir = root_dir
        self.target_length = target_length
        self.sr = sr
        
        # Define the 35 classes (30 commands + silence + unknown + '_background_noise_')
        self.classes = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 
            'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 
            'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
            'backward', 'begin', 'bird', 'cat', 'dog', 'eat', 'end', 'follow', 'forward',
            'learn', 'marvin', 'off', 'on', 'right', 'sheila', 'stop', 'up', 'visual', 'word', 'zero'
        ]
        # Remove duplicates and add special classes
        self.classes = sorted(list(set([
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'bed', 'bird', 'cat', 'dog', 'down', 'follow', 'forward', 'go', 'happy', 'house',
            'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'right', 'seven', 'sheila',
            'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'backward',
            'begin', 'bird', 'cat', 'dog', 'eat', 'end', 'follow', 'forward', 'learn', 'marvin',
            'off', 'on', 'right', 'sheila', 'stop', 'up', 'visual', 'word',
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
            'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
            'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
            'wow', 'yes', 'zero'
        ])))
        
        # Add special classes
        self.classes = ['_silence_', '_unknown_'] + sorted(list(set([
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
            'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
            'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
            'wow', 'yes', 'zero', 'backward', 'begin', 'bird', 'cat', 'dog', 'eat', 'end', 'follow',
            'forward', 'learn', 'marvin', 'off', 'on', 'right', 'sheila', 'stop', 'up', 'visual', 'word'
        ])))
        
        # Actual SCV2 classes
        self.classes = ['_silence_', '_unknown_'] + [
            'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
            'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
            'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
            'wow', 'yes', 'zero'
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build file list based on subset
        self.file_list = []
        self.labels = []
        
        if subset == 'validation':
            # Load validation list from official file
            validation_list_path = os.path.join(root_dir, 'validation_list.txt')
            if os.path.exists(validation_list_path):
                with open(validation_list_path, 'r') as f:
                    validation_files = [line.strip() for line in f.readlines()]
                self._build_file_list(validation_files)
            else:
                # If validation file doesn't exist, use a heuristic
                self._build_subset_list(subset)
        elif subset == 'testing':
            # Load testing list from official file
            testing_list_path = os.path.join(root_dir, 'testing_list.txt')
            if os.path.exists(testing_list_path):
                with open(testing_list_path, 'r') as f:
                    testing_files = [line.strip() for line in f.readlines()]
                self._build_file_list(testing_files)
            else:
                # If testing file doesn't exist, use a heuristic
                self._build_subset_list(subset)
        else:  # training
            # Load training list by excluding validation and testing
            validation_list_path = os.path.join(root_dir, 'validation_list.txt')
            testing_list_path = os.path.join(root_dir, 'testing_list.txt')
            
            validation_files = set()
            testing_files = set()
            
            if os.path.exists(validation_list_path):
                with open(validation_list_path, 'r') as f:
                    validation_files = set(line.strip() for line in f.readlines())
            if os.path.exists(testing_list_path):
                with open(testing_list_path, 'r') as f:
                    testing_files = set(line.strip() for line in f.readlines())
            
            self._build_training_list(validation_files, testing_files)
    
    def _build_file_list(self, file_paths):
        """Build file list from specific file paths."""
        for file_path in file_paths:
            full_path = os.path.join(self.root_dir, file_path)
            if os.path.exists(full_path):
                class_name = file_path.split('/')[0]  # Directory name is the class
                label = self.class_to_idx.get(class_name, 1)  # Unknown class for missing labels
                self.file_list.append(full_path)
                self.labels.append(label)
    
    def _build_subset_list(self, subset):
        """Build subset list using heuristics."""
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Skip special directories
            if class_name.startswith('_') and class_name.endswith('_'):
                continue
                
            class_idx = self.class_to_idx.get(class_name, 1)  # Unknown class for unrecognized
            
            # Get all WAV files in class directory
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            # Use heuristic to split files (this is approximate)
            n_files = len(wav_files)
            if subset == 'train':
                start_idx, end_idx = 0, int(0.8 * n_files)
            elif subset == 'validation':
                start_idx = int(0.8 * n_files)
                end_idx = int(0.9 * n_files)
            else:  # testing
                start_idx = int(0.9 * n_files)
                end_idx = n_files
            
            selected_files = wav_files[start_idx:end_idx]
            
            for file_name in selected_files:
                full_path = os.path.join(class_path, file_name)
                self.file_list.append(full_path)
                self.labels.append(class_idx)
    
    def _build_training_list(self, validation_files, testing_files):
        """Build training list by excluding validation and testing files."""
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Skip special directories
            if class_name.startswith('_') and class_name.endswith('_'):
                continue
                
            class_idx = self.class_to_idx.get(class_name, 1)  # Unknown class for unrecognized
            
            # Get all WAV files in class directory
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            
            for file_name in wav_files:
                rel_path = os.path.join(class_name, file_name)
                if rel_path in validation_files or rel_path in testing_files:
                    continue  # Skip if in validation or testing
                    
                full_path = os.path.join(class_path, file_name)
                self.file_list.append(full_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (spectrogram, label) where label is the class index
        """
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            waveform, orig_sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if orig_sr != self.sr:
                resampler = T.Resample(orig_sr, self.sr)
                waveform = resampler(waveform)
            
            # Pad or crop to 1 second (required for consistent processing)
            target_samples = int(self.sr * 1.0)  # 16000 samples at 16kHz for 1 sec
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
            waveform = torch.zeros(1, int(self.sr * 1.0))
        
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
        
        # Pad or trim to target length (adjust for 1 sec audio)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            # Instead of zero padding, repeat the audio content to fill the space
            # This helps maintain meaningful content instead of padding with zeros
            if n_frames > 0:  # Only repeat if we have content
                # Calculate how many times to repeat the frames
                repeats_needed = (self.target_length + n_frames - 1) // n_frames  # Ceiling division
                fbank = fbank.repeat(repeats_needed, 1)[:self.target_length, :]  # Repeat and trim to target
            else:
                # If somehow fbank is empty, use zero padding as fallback
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:self.target_length, :]
        
        # Normalize as done in the original implementation
        fbank = (fbank - (-4.26)) / (4.56 * 2)
        
        # Add channel dimension
        fbank = fbank.unsqueeze(0)
        
        return fbank, label


def get_sc2_stats():
    """Print statistics about Speech Commands V2 dataset."""
    print("📊 Speech Commands V2 Dataset Statistics:")
    print("- 105,829 audio files")
    print("- 35 classes (30 commands + silence + unknown + background noise)")
    print("- ~1 second audio clips")
    print("- Classes: zero, one, two, ..., up, down, left, right, yes, no, etc.")


if __name__ == "__main__":
    # Example usage
    get_sc2_stats()