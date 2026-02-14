import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from tqdm import tqdm
import numpy as np
import random
from models.audiomae import mae_vit_base_patch16
from synthesizer.world_model import PhysicsBasedDataset


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main pretraining function with on-the-fly data generation."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Pre-training with Advanced Physics Synth on {device}")
    
    # Model configuration (matching the paper specifications)
    model = mae_vit_base_patch16(
        norm_pix_loss=True, 
        audio_exp=True, 
        in_chans=1, 
        img_size=(1024, 128),  # Log-Mel spectrogram dimensions: 1024 time frames x 128 mel bins
        stride=16  # Patch stride for non-overlapping patches
    )
    model.to(device)
    
    # Optimizer configuration (based on the 2.3 implementation)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        betas=(0.9, 0.95), 
        weight_decay=0.05
    )
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # Dataset and DataLoader - On-the-fly generation
    print("🔄 Creating physics-based dataset with on-the-fly generation...")
    dataset = PhysicsBasedDataset(
        epoch_len=5000,  # Number of samples per epoch (as in 2.3 implementation)
        sr=16000,        # Sample rate (16kHz as per paper)
        duration=10.24   # Duration (10.24s as per paper)
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=16,      # Batch size (as in 2.3 implementation)
        num_workers=4,      # Number of workers for data loading
        pin_memory=True,    # Pin memory for faster GPU transfer
        shuffle=True        # Shuffle for better training
    )
    
    # Training parameters
    EPOCHS = 500  # As specified in the paper for full training
    print(f"🔥 Starting Training for {EPOCHS} Epochs...")
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📈 Batch Size: {dataloader.batch_size}, Steps per Epoch: {len(dataloader)}")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                loss, _, _ = model(images, mask_ratio=0.75)  # 75% masking as per paper
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/transaudio_base_ep{epoch+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Early stopping check (optional)
        if avg_loss < 1e-6:  # Very low loss indicates possible convergence
            print("🎯 Loss converged, stopping early...")
            break

    # Final save
    final_checkpoint = "checkpoints/transaudio_base_final.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_checkpoint)
    
    print(f"🏁 Pre-training Complete!")
    print(f"💾 Final model saved to: {final_checkpoint}")
    print(f"📈 Final loss: {avg_loss:.6f}")


def validate_model_loading():
    """Validate that the model can be loaded correctly."""
    print("🔍 Validating model loading...")
    
    # Create model
    model = mae_vit_base_patch16(
        norm_pix_loss=True, 
        audio_exp=True, 
        in_chans=1, 
        img_size=(1024, 128)
    )
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 1, 1024, 128)  # Batch of 2 spectrograms
    model.eval()
    
    with torch.no_grad():
        loss, pred, mask = model(dummy_input, mask_ratio=0.75)
    
    print(f"✅ Model validation passed!")
    print(f"📊 Input shape: {dummy_input.shape}")
    print(f"📊 Prediction shape: {pred.shape}")
    print(f"📊 Loss: {loss.item():.4f}")
    

if __name__ == '__main__':
    # Validate model first
    validate_model_loading()
    print()
    
    # Run main training
    main()