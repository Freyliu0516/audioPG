import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from models.audiomae import AudioMAEClassifier
from dataset.esc50 import ESC50Dataset
from dataset.us8k import UrbanSound8KDataset
from dataset.speechcommands import SpeechCommandsV2Dataset


def finetune_esc50(pretrained_ckpt, data_path, num_epochs=100, batch_size=32, lr=2e-4):
    """Fine-tune on ESC-50 dataset."""
    print("🎵 Fine-tuning on ESC-50...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model for classification
    model = AudioMAEClassifier(
        num_classes=50,           # ESC-50 has 50 classes
        drop_path_rate=0.1,       # Stochastic depth for regularization
        stride=16                 # Match pre-trained model
    )
    
    # Load pre-trained weights
    if os.path.exists(pretrained_ckpt):
        print(f"📥 Loading pre-trained weights from {pretrained_ckpt}...")
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        
        # Handle both full checkpoints and model-only checkpoints
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Filter out decoder-related weights (not needed for classification)
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not any(skip_key in k for skip_key in ['decoder', 'mask_token', 'decoder_pred']):
                filtered_state_dict[k] = v
                
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Pre-trained weights loaded successfully!")
    else:
        print(f"⚠️ Warning: Pre-trained checkpoint {pretrained_ckpt} not found. Training from scratch.")
    
    model.to(device)
    
    # Mixup for regularization
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=50
    )
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.05
    )
    
    # Loss functions
    criterion_train = SoftTargetCrossEntropy()
    criterion_val = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Data loaders
    train_dataset = ESC50Dataset(data_path, fold=1, train=True)  # Using fold 1 for demo
    val_dataset = ESC50Dataset(data_path, fold=1, train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"ESC-50 Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            # Apply mixup
            if mixup_fn is not None:
                data, targets = mixup_fn(data, targets)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion_train(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/esc50_best.pth')
            print(f"💾 New best model saved with accuracy: {best_acc:.2f}%")
    
    print(f"✅ ESC-50 fine-tuning completed. Best accuracy: {best_acc:.2f}%")
    return best_acc


def finetune_us8k(pretrained_ckpt, data_path, num_epochs=100, batch_size=32, lr=2e-4):
    """Fine-tune on UrbanSound8K dataset."""
    print("🏙️ Fine-tuning on UrbanSound8K...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model for classification
    model = AudioMAEClassifier(
        num_classes=10,           # US8K has 10 classes
        drop_path_rate=0.1,       # Stochastic depth for regularization
        stride=16                 # Match pre-trained model
    )
    
    # Load pre-trained weights
    if os.path.exists(pretrained_ckpt):
        print(f"📥 Loading pre-trained weights from {pretrained_ckpt}...")
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        
        # Handle both full checkpoints and model-only checkpoints
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Filter out decoder-related weights (not needed for classification)
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not any(skip_key in k for skip_key in ['decoder', 'mask_token', 'decoder_pred']):
                filtered_state_dict[k] = v
                
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Pre-trained weights loaded successfully!")
    else:
        print(f"⚠️ Warning: Pre-trained checkpoint {pretrained_ckpt} not found. Training from scratch.")
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.05
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Data loaders
    train_dataset = UrbanSound8KDataset(data_path, fold=1, train=True)  # Using fold 1 for demo
    val_dataset = UrbanSound8KDataset(data_path, fold=1, train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"US8K Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/us8k_best.pth')
            print(f"💾 New best model saved with accuracy: {best_acc:.2f}%")
    
    print(f"✅ UrbanSound8K fine-tuning completed. Best accuracy: {best_acc:.2f}%")
    return best_acc


def finetune_speech_commands(pretrained_ckpt, data_path, num_epochs=50, batch_size=32, lr=2e-4):
    """Fine-tune on Speech Commands V2 dataset."""
    print("🗣️ Fine-tuning on Speech Commands V2...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataset to get the correct number of classes
    train_dataset = SpeechCommandsV2Dataset(data_path, subset='train')
    
    # Initialize model for classification
    model = AudioMAEClassifier(
        num_classes=len(train_dataset.classes),  # Dynamically get number of classes
        drop_path_rate=0.1,       # Stochastic depth for regularization
        stride=16                 # Match pre-trained model
    )
    
    # Load pre-trained weights
    if os.path.exists(pretrained_ckpt):
        print(f"📥 Loading pre-trained weights from {pretrained_ckpt}...")
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        
        # Handle both full checkpoints and model-only checkpoints
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Filter out decoder-related weights (not needed for classification)
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not any(skip_key in k for skip_key in ['decoder', 'mask_token', 'decoder_pred']):
                filtered_state_dict[k] = v
                
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ Pre-trained weights loaded successfully!")
    else:
        print(f"⚠️ Warning: Pre-trained checkpoint {pretrained_ckpt} not found. Training from scratch.")
    
    model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.05
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Data loaders
    train_dataset = SpeechCommandsV2Dataset(data_path, subset='train')
    val_dataset = SpeechCommandsV2Dataset(data_path, subset='validation')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"SCV2 Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/scv2_best.pth')
            print(f"💾 New best model saved with accuracy: {best_acc:.2f}%")
    
    print(f"✅ Speech Commands V2 fine-tuning completed. Best accuracy: {best_acc:.2f}%")
    return best_acc


def run_finetuning_suite(pretrained_ckpt="checkpoints/transaudio_base_final.pth"):
    """Run fine-tuning on all downstream tasks."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    print("🚀 Starting fine-tuning suite for all downstream tasks...")
    
    # Define paths (these should be configured by the user)
    esc50_path = "/path/to/esc-50"  # User needs to set this
    us8k_path = "/path/to/urbansound8k"  # User needs to set this
    scv2_path = "/path/to/speech_commands_v2"  # User needs to set this
    
    results = {}
    
    # Fine-tune on ESC-50
    if os.path.exists(esc50_path):
        esc50_acc = finetune_esc50(pretrained_ckpt, esc50_path)
        results['ESC-50'] = esc50_acc
    else:
        print("⚠️ ESC-50 path not found, skipping...")
    
    # Fine-tune on UrbanSound8K
    if os.path.exists(us8k_path):
        us8k_acc = finetune_us8k(pretrained_ckpt, us8k_path)
        results['UrbanSound8K'] = us8k_acc
    else:
        print("⚠️ UrbanSound8K path not found, skipping...")
    
    # Fine-tune on Speech Commands V2
    if os.path.exists(scv2_path):
        scv2_acc = finetune_speech_commands(pretrained_ckpt, scv2_path)
        results['SpeechCommandsV2'] = scv2_acc
    else:
        print("⚠️ Speech Commands V2 path not found, skipping...")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 Fine-tuning Results Summary")
    print("="*50)
    for dataset, acc in results.items():
        print(f"{dataset:<20}: {acc:.2f}%")
    
    if results:
        avg_acc = sum(results.values()) / len(results)
        print("-"*50)
        print(f"{'Average Accuracy':<20}: {avg_acc:.2f}%")
        print("="*50)
    
    return results


def evaluate_model(pretrained_ckpt, dataset_name, data_path):
    """Evaluate a pre-trained model on a specific dataset."""
    print(f"🔍 Evaluating model on {dataset_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine number of classes based on dataset
    if dataset_name.lower() == 'esc-50':
        num_classes = 50
    elif dataset_name.lower() == 'urbansound8k':
        num_classes = 10
    elif dataset_name.lower() == 'speechcommands':
        num_classes = 35
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize model
    model = AudioMAEClassifier(
        num_classes=num_classes,
        drop_path_rate=0.0,  # No dropout during evaluation
        stride=16
    )
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Filter out decoder weights
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if not any(skip_key in k for skip_key in ['decoder', 'mask_token', 'decoder_pred']):
            filtered_state_dict[k] = v
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Select appropriate dataset
    if dataset_name.lower() == 'esc-50':
        dataset = ESC50Dataset(data_path, fold=5, train=False)  # Use fold 5 for validation
    elif dataset_name.lower() == 'urbansound8k':
        dataset = UrbanSound8KDataset(data_path, fold=10, train=False)  # Use fold 10 for validation
    elif dataset_name.lower() == 'speechcommands':
        dataset = SpeechCommandsV2Dataset(data_path, subset='testing')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"✅ Evaluation completed. Accuracy on {dataset_name}: {accuracy:.2f}%")
    
    return accuracy


if __name__ == '__main__':
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run the fine-tuning suite
    results = run_finetuning_suite()