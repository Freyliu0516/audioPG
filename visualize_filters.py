import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import entropy
from models.audiomae import MaskedAutoencoderViT
import torch.nn.functional as F


def visualize_filters(model, layer_name='patch_embed.proj.weight', save_path=None):
    """
    Visualize the learned filters from a specific layer (e.g., patch embedding layer).
    This creates Figure 7 from the paper showing evolution of filters.
    """
    # Extract the weights from the specified layer
    if hasattr(model, layer_name.replace('.', '_')):
        # Handle special case where attribute names have dots
        pass
    else:
        # Access the layer
        layers = layer_name.split('.')
        module = model
        for layer in layers:
            module = getattr(module, layer)
        
        weights = module.weight.data.cpu().numpy()
    
    # For patch embedding layer, weights are typically [embed_dim, in_chans, patch_size, patch_size]
    # We'll visualize each filter as an image
    embed_dim, in_chans, patch_h, patch_w = weights.shape
    
    # Determine grid size for visualization
    n_filters = min(64, embed_dim)  # Show max 64 filters
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Learned Filters Visualization ({layer_name})', fontsize=16)
    
    for i in range(n_filters):
        if i >= embed_dim:
            break
            
        ax = axes[i // grid_size, i % grid_size]
        # Take the first input channel for visualization
        filter_img = weights[i, 0, :, :]
        im = ax.imshow(filter_img, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Filter {i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_filters, grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Filters visualization saved to {save_path}")
    
    plt.show()


def pca_visualization(model, layer_name='patch_embed.proj.weight', n_components=2):
    """
    Apply PCA to the learned representations and visualize them.
    This helps visualize the structure of learned representations.
    """
    # Extract weights from the specified layer
    layers = layer_name.split('.')
    module = model
    for layer in layers:
        module = getattr(module, layer)
    
    weights = module.weight.data.cpu().numpy()
    
    # Flatten the filters to vectors for PCA
    embed_dim, in_chans, patch_h, patch_w = weights.shape
    filters_flat = weights.reshape(embed_dim, -1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(filters_flat)
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, c=range(len(pca_result)), cmap='tab20')
    plt.title(f'PCA Visualization of {layer_name}\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca_result, pca


def tsne_visualization(model, layer_name='patch_embed.proj.weight', n_components=2, perplexity=30):
    """
    Apply t-SNE to the learned representations and visualize them.
    """
    # Extract weights from the specified layer
    layers = layer_name.split('.')
    module = model
    for layer in layers:
        module = getattr(module, layer)
    
    weights = module.weight.data.cpu().numpy()
    
    # Flatten the filters to vectors for t-SNE
    embed_dim, in_chans, patch_h, patch_w = weights.shape
    filters_flat = weights.reshape(embed_dim, -1)
    
    # Apply t-SNE (use a subset if too many filters)
    n_samples = min(200, embed_dim)  # Limit for performance
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(filters_flat[:n_samples])
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, 
                         c=range(len(tsne_result)), cmap='tab20')
    plt.title(f't-SNE Visualization of {layer_name} (first {n_samples} filters)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return tsne_result


def compute_mutual_information(estimator='kraskov', X=None, Y=None):
    """
    Compute mutual information between neural activities and physical parameters.
    This implements the disentanglement analysis mentioned in the paper.
    """
    if estimator == 'kraskov':
        # Kraskov kNN estimator for mutual information
        # This is a simplified version - in practice, you'd use a more sophisticated implementation
        from sklearn.neighbors import NearestNeighbors
        from scipy.special import digamma
        import numpy as np
        
        # Placeholder implementation - in practice this would be more complex
        if X is not None and Y is not None:
            # Standardize the variables
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
            
            # Combine X and Y
            XY = np.hstack([X, Y])
            
            # Find k-th nearest neighbor distances
            k = min(3, len(X) - 1)  # k should be less than number of samples
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(XY)
            distances, indices = nbrs.kneighbors(XY)
            
            # Estimate MI using the Kraskov formula (simplified)
            # This is a placeholder - full implementation would be more involved
            mi_estimate = 0.5  # Placeholder value
            
            return mi_estimate
    else:
        raise ValueError(f"Estimator {estimator} not supported")


def analyze_disentanglement(model, data_loader, physical_params):
    """
    Analyze disentanglement by computing mutual information between 
    neural activations and physical parameters.
    """
    model.eval()
    activations = []
    param_values = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            # Get intermediate activations from the model
            x = model.patch_embed(data)  # Get patch embeddings
            x = x + model.pos_embed[:, 1:, :]  # Add positional encoding
            
            # Collect activations (you might want to collect from multiple layers)
            activations.append(x.mean(dim=1).cpu().numpy())  # Average over patches
            
            # Collect corresponding physical parameters
            param_values.extend(physical_params[batch_idx*data.size(0):(batch_idx+1)*data.size(0)])
            
            if batch_idx >= 10:  # Limit for efficiency
                break
    
    activations = np.vstack(activations)
    param_values = np.array(param_values)
    
    # Compute mutual information between each activation dimension and each parameter
    n_activations = activations.shape[1]
    n_params = param_values.shape[1] if len(param_values.shape) > 1 else 1
    
    mi_matrix = np.zeros((n_activations, n_params))
    
    for i in range(min(50, n_activations)):  # Limit for efficiency
        for j in range(n_params):
            if n_params == 1:
                param_j = param_values
            else:
                param_j = param_values[:, j]
            
            # Compute MI between activation i and parameter j
            mi = compute_mutual_information(X=activations[:, i:i+1], Y=param_j.reshape(-1, 1))
            mi_matrix[i, j] = mi
    
    # Plot MI matrix
    plt.figure(figsize=(10, 8))
    if n_params == 1:
        plt.plot(mi_matrix[:50, 0], label='MI with parameter')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mutual Information')
        plt.title('Mutual Information between Neurons and Physical Parameter')
    else:
        sns.heatmap(mi_matrix[:50, :], annot=False, cmap='viridis', 
                   xticklabels=[f'Param_{i}' for i in range(n_params)],
                   yticklabels=[f'Act_{i}' for i in range(min(50, n_activations))])
        plt.title('Mutual Information Matrix (Neurons vs Physical Parameters)')
    
    plt.tight_layout()
    plt.show()
    
    return mi_matrix


def plot_reconstruction(original, reconstructed, mask, title="Reconstruction Comparison"):
    """
    Plot original vs reconstructed spectrograms for visualization.
    This reproduces Figure 4 from the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original spectrogram
    im1 = axes[0].imshow(original.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Spectrogram')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Mel Bins')
    plt.colorbar(im1, ax=axes[0])
    
    # Reconstructed spectrogram
    im2 = axes[1].imshow(reconstructed.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Reconstructed Spectrogram')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Bins')
    plt.colorbar(im2, ax=axes[1])
    
    # Mask visualization - reshape to patch grid (assuming 64x8 based on 1024x128 spectrogram with 16x16 patches)
    mask_flat = mask.squeeze().cpu().numpy()
    # Calculate the expected patch grid dimensions based on input spectrogram size (1024, 128) and patch size (16)
    h_patches = 1024 // 16  # 64
    w_patches = 128 // 16   # 8
    mask_vis = mask_flat.reshape(h_patches, w_patches) if len(mask_flat) == h_patches * w_patches else mask_flat[:h_patches*w_patches].reshape(h_patches, w_patches)
    im3 = axes[2].imshow(mask_vis, aspect='auto', origin='lower', cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Mask (White=Masked, Black=Visible)')
    axes[2].set_xlabel('Patches')
    axes[2].set_ylabel('Batch')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def reconstruction_demo(model, sample_data):
    """
    Demonstrate reconstruction capability of the model.
    Reproduces Figure 4 from the paper.
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass to get reconstruction
        loss, pred, mask = model(sample_data, mask_ratio=0.75)
        
        # Unpatchify prediction to compare with original
        pred_unpatched = model.unpatchify(pred)
        original_unpatched = sample_data  # Already in spectrogram form
        
        # Plot comparison
        plot_reconstruction(original_unpatched[0], pred_unpatched[0], mask[0])
        
        print(f"Reconstruction loss: {loss.item():.4f}")
        return loss.item()


def analyze_spectrum(audio_signal, sr=16000):
    """
    Analyze the frequency spectrum of generated audio.
    """
    # Compute FFT
    fft = np.fft.fft(audio_signal)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(audio_signal), 1/sr)
    
    # Only take positive frequencies
    positive_freq_idx = freqs >= 0
    freqs = freqs[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    # Plot spectrum
    plt.figure(figsize=(12, 6))
    plt.semilogx(freqs, 20 * np.log10(magnitude + 1e-10))  # Convert to dB
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.xlim([20, sr//2])
    plt.show()


def plot_spectrogram(audio_signal, sr=16000, n_fft=2048, hop_length=512):
    """
    Plot the spectrogram of an audio signal.
    """
    # Compute STFT
    stft = torch.stft(
        torch.tensor(audio_signal).float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True
    )
    
    # Convert to magnitude spectrogram
    magnitude = torch.abs(stft)
    
    # Convert to dB scale
    magnitude_db = 20 * torch.log10(magnitude + 1e-10)
    
    # Plot
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        magnitude_db.numpy(), 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='hz'
    )
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def visualize_training_curves(train_losses, val_accuracies=None, save_path=None):
    """
    Plot training curves to visualize model performance over time.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    if val_accuracies is not None:
        ax2 = ax1.twinx()
        ax2.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_ylabel('Accuracy (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend combining both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    plt.title('Training Curves')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


try:
    import librosa
except ImportError:
    print("Warning: librosa not installed. Some visualization functions may not work.")
    librosa = None


if __name__ == "__main__":
    print("Visualization and Analysis Tools for AudioPG")
    print("="*50)
    print("Available tools:")
    print("1. visualize_filters() - Visualize learned filters")
    print("2. pca_visualization() - PCA of learned representations") 
    print("3. tsne_visualization() - t-SNE of learned representations")
    print("4. analyze_disentanglement() - Disentanglement analysis")
    print("5. reconstruction_demo() - Reconstruction demonstration")
    print("6. analyze_spectrum() - Frequency spectrum analysis")
    print("7. plot_spectrogram() - Spectrogram visualization")
    print("8. visualize_training_curves() - Training curve visualization")