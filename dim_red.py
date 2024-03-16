import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error


def perform_pca(dataset, x, rs):
    pca = PCA(random_state=rs)
    pca.fit(x)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot eigenvalues
    eigenvalues = pca.explained_variance_
    axes[0].plot(range(1, pca.n_components_ + 1), eigenvalues, marker='o')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Eigenvalues')
    axes[0].set_title('Scree Plot')
    for i, (x, y) in enumerate(zip(range(1, pca.n_components_ + 1), eigenvalues)):
        axes[0].annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(3,10), ha='center')

    # Plot singular values
    singular_values = pca.singular_values_
    axes[1].plot(range(1, pca.n_components_ + 1), singular_values, marker='o')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Singular Values')
    axes[1].set_title('Singular Values')
    for i, (x, y) in enumerate(zip(range(1, pca.n_components_ + 1), singular_values)):
        axes[1].annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(3,10), ha='center')

    # Plot cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    axes[2].plot(range(1, pca.n_components_ + 1), cumulative_variance, marker='o')
    axes[2].set_xlabel('Number of Components')
    axes[2].set_ylabel('Cumulative Explained Variance')
    axes[2].set_title('Explained Variance')
    for i, (x, y) in enumerate(zip(range(1, pca.n_components_ + 1), cumulative_variance)):
        axes[2].annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(-3,10), ha='center')

    plt.tight_layout()
    plt.savefig(f'Images/{dataset}/PCA/Component Analysis.png')
    plt.close()

def perform_ica(dataset, x, rs):
    kurtosis_values = []
    max_components = x.shape[1]

    for n_components in range(1, max_components + 1):
        ica = FastICA(n_components=n_components, random_state=rs)
        x_transformed = ica.fit_transform(x)
        kurtosis_values.append(np.mean(kurtosis(x_transformed)))

    # Plot kurtosis values
    plt.plot(range(1, max_components + 1), kurtosis_values, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Kurtosis')
    plt.title('Kurtosis vs. Number of Components')
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/ICA/Component Analysis.png')
    plt.close()

def perform_rp(dataset, x, rs):
    reconstruction_errors = []
    max_components = x.shape[1]

    # Apply Random Projection and compute reconstruction error for each number of components
    for components_removed in range(0, max_components):
        rp = GaussianRandomProjection(n_components=max_components - components_removed, random_state=rs, compute_inverse_components=True)
        x_projected = rp.fit_transform(x)
        x_reconstructed = rp.inverse_transform(x_projected)
        reconstruction_error = mean_squared_error(x, x_reconstructed)
        reconstruction_errors.append(reconstruction_error)

    # Plot reconstruction error
    plt.plot(range(1, max_components + 1), reconstruction_errors, marker='o')
    plt.xlabel('Components Removed')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs. Components Removed')
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/RP/Component Analysis.png')
    plt.close()
    return

def perform_isomap(dataset, x):
    reconstruction_errors = []
    components_range = range(0, 21)

    # Loop over each number of components and evaluate the model
    for components_removed in components_range:
        isomap = Isomap(n_components=21 - components_removed, n_neighbors=10)
        isomap.fit_transform(x)
        reconstruction_error = isomap.reconstruction_error()
        reconstruction_errors.append(reconstruction_error)

    # Plot the evaluation metric vs. the number of components
    plt.plot(components_range, reconstruction_errors, marker='o')
    plt.xlabel('Components Removed')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs. Components Removed')
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/Isomap/Component Analysis.png')
    plt.close()

    return