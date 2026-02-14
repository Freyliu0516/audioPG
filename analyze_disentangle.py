import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns


def kraskov_mi_estimator(X, Y, k=3):
    """
    Kraskov kNN estimator for mutual information between continuous variables X and Y.
    
    Args:
        X: Array of shape (n_samples, n_features_x)
        Y: Array of shape (n_samples, n_features_y) 
        k: Number of nearest neighbors to use
    
    Returns:
        Estimated mutual information
    """
    # Ensure inputs are 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_samples = X.shape[0]
    
    # Combine X and Y
    XY = np.hstack([X, Y])
    
    # Fit kNN on the joint space
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(XY)
    distances, indices = nbrs.kneighbors(XY)
    
    # Use the distance to the kth nearest neighbor
    epsilon = distances[:, k]  # kth nearest neighbor distance
    
    # Count neighbors within epsilon in marginal spaces
    nbrs_x = NearestNeighbors(radius=epsilon[:, None]).fit(X)
    n_x = nbrs_x.radius_neighbors(X, return_distance=False)
    nx = np.array([len(neighbors) for neighbors in n_x]) - 1  # Subtract 1 to exclude self
    
    nbrs_y = NearestNeighbors(radius=epsilon[:, None]).fit(Y)
    n_y = nbrs_y.radius_neighbors(Y, return_distance=False)
    ny = np.array([len(neighbors) for neighbors in n_y]) - 1  # Subtract 1 to exclude self
    
    # Calculate MI estimate
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n_samples)
    return max(0, mi)  # Mutual information should be non-negative


def analyze_neuron_disentanglement(model, phys_params, activations):
    """
    Analyze how well neurons encode individual physical parameters (disentanglement).
    
    Args:
        model: Trained model
        phys_params: Physical parameters used during generation [n_samples, n_params]
        activations: Neural activations [n_samples, n_neurons]
    
    Returns:
        MI matrix [n_neurons, n_params]
    """
    n_neurons = activations.shape[1]
    n_params = phys_params.shape[1]
    
    # Compute MI between each neuron and each parameter
    mi_matrix = np.zeros((n_neurons, n_params))
    
    for i in range(min(100, n_neurons)):  # Limit for efficiency
        for j in range(n_params):
            mi = kraskov_mi_estimator(
                activations[:, i].reshape(-1, 1), 
                phys_params[:, j].reshape(-1, 1)
            )
            mi_matrix[i, j] = mi
    
    return mi_matrix


def plot_disentanglement_analysis(mi_matrix, param_names=None):
    """
    Plot the disentanglement analysis results.
    """
    n_neurons, n_params = mi_matrix.shape
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    if param_names is None:
        param_names = [f'Param_{i}' for i in range(n_params)]
    
    neuron_labels = [f'Neuron_{i}' for i in range(min(50, n_neurons))]  # Show max 50 neurons
    
    # Truncate matrix for visualization if too large
    mi_truncated = mi_matrix[:min(50, n_neurons), :]
    
    sns.heatmap(
        mi_truncated,
        xticklabels=param_names,
        yticklabels=neuron_labels,
        annot=False,
        cmap='viridis',
        cbar_kws={'label': 'Mutual Information'}
    )
    
    plt.title('Neuron-Parameter Mutual Information (Disentanglement Analysis)')
    plt.xlabel('Physical Parameters')
    plt.ylabel('Neurons')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compute_disentanglement_score(mi_matrix):
    """
    Compute disentanglement score based on MI matrix.
    A perfectly disentangled representation would have each neuron encoding only one parameter.
    """
    # Normalize MI matrix rows to sum to 1 (each neuron's information distribution)
    mi_norm = mi_matrix / (mi_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    # Compute entropy of each neuron's parameter distribution
    # Low entropy means the neuron focuses on few parameters (good for disentanglement)
    entropies = -np.sum(mi_norm * np.log(mi_norm + 1e-10), axis=1)
    
    # Disentanglement score: average of how concentrated each neuron's information is
    # Higher scores indicate better disentanglement (each neuron encodes few parameters)
    disent_scores = 1 - (entropies / np.log(mi_matrix.shape[1]))  # Normalize by max entropy
    avg_disentanglement = np.mean(disent_scores)
    
    return avg_disentanglement, disent_scores


def synthetic_experiment():
    """
    Run a synthetic experiment to demonstrate disentanglement analysis.
    """
    print("🧪 Running synthetic disentanglement analysis...")
    
    # Simulate physical parameters
    n_samples = 1000
    n_params = 5  # e.g., fundamental frequency, decay rate, etc.
    phys_params = np.random.randn(n_samples, n_params)
    
    # Simulate neural activations
    n_neurons = 128
    activations = np.random.randn(n_samples, n_neurons)
    
    # Add some dependency between parameters and activations to make it interesting
    for i in range(n_params):
        # Each parameter influences a subset of neurons
        neuron_subset = np.random.choice(n_neurons, size=20, replace=False)
        activations[:, neuron_subset] += phys_params[:, i:i+1] * 0.5
    
    # Analyze disentanglement
    mi_matrix = analyze_neuron_disentanglement(None, phys_params, activations)
    
    # Plot results
    param_names = ['Fundamental Freq', 'Decay Rate', 'Modulation Index', 
                   'Noise Level', 'Filter Cutoff']
    plot_disentanglement_analysis(mi_matrix, param_names)
    
    # Compute disentanglement score
    avg_disent, disent_scores = compute_disentanglement_score(mi_matrix)
    print(f"Average disentanglement score: {avg_disent:.3f}")
    print(f"Range: [{disent_scores.min():.3f}, {disent_scores.max():.3f}]")
    
    return mi_matrix, avg_disent


def analyze_real_model_disentanglement(model, data_loader, param_extractor_func):
    """
    Analyze disentanglement in a real trained model.
    
    Args:
        model: Trained AudioPG model
        data_loader: DataLoader with generated samples
        param_extractor_func: Function to extract physical parameters from samples
    """
    model.eval()
    all_activations = []
    all_params = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            # Get intermediate activations (e.g., from encoder)
            x = model.patch_embed(data)
            x = x + model.pos_embed[:, 1:, :]
            
            # Get encoded representations
            for blk in model.blocks:
                x = blk(x)
            x = model.norm(x)
            
            # Use CLS token representation or average pooling
            encoded_repr = x[:, 0, :]  # Use CLS token
            
            # Extract physical parameters from data
            batch_params = param_extractor_func(data)
            
            all_activations.append(encoded_repr.cpu().numpy())
            all_params.append(batch_params)
            
            if batch_idx >= 20:  # Limit for efficiency
                break
    
    all_activations = np.vstack(all_activations)
    all_params = np.vstack(all_params)
    
    # Analyze disentanglement
    mi_matrix = analyze_neuron_disentanglement(model, all_params, all_activations)
    
    # Compute disentanglement metrics
    avg_disent, _ = compute_disentanglement_score(mi_matrix)
    
    print(f"Real model disentanglement score: {avg_disent:.3f}")
    
    return mi_matrix, avg_disent


if __name__ == "__main__":
    print("Disentanglement Analysis for AudioPG")
    print("="*40)
    
    # Run synthetic experiment
    mi_matrix, disent_score = synthetic_experiment()
    
    print("\n💡 Key Metrics:")
    print(f"- MI Matrix Shape: {mi_matrix.shape}")
    print(f"- Disentanglement Score: {disent_score:.3f} (higher is better)")
    print("- Perfect disentanglement ≈ 1.0")
    print("- Random encoding ≈ 0.0")
    
    print("\n📋 Interpretation:")
    print("Each cell in the heatmap represents the mutual information between")
    print("a neuron and a physical parameter. Well-disentangled representations")
    print("will show neurons that strongly encode only specific parameters.")