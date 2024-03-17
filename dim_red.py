import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error


def perform_pca(dataset, x, rs):
    eigenvalues = []
    max_components = x.shape[1]
    components_range = range(1, max_components+1)

    for n_components in components_range:
        pca = PCA(n_components=n_components, random_state=rs)
        pca.fit_transform(x)
        eigenvalues.append(np.mean(pca.explained_variance_))

    # Plot both curves on the same graph
    plt.figure(figsize=(10, 5))

    # Plot eigenvalues
    plt.plot(range(1, pca.n_components_ + 1), eigenvalues, marker='o', label='Eigenvalues')
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalues')
    plt.title('Scree Plot')
    for x, y in zip(range(1, pca.n_components_ + 1), eigenvalues):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(5,5), ha='center', weight='bold', color='blue')

    # Plot cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, pca.n_components_ + 1), cumulative_variance, marker='o', label='Cumulative Explained Variance')
    for x, y in zip(range(1, pca.n_components_ + 1), cumulative_variance):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(-5,5), ha='center', weight='bold', color='darkorange')

    plt.xticks(range(1, pca.n_components_ + 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Images/{dataset}/PCA/Component Analysis.png')
    plt.close()

    return np.where(cumulative_variance >= 0.8)[0][0]

def perform_ica(dataset, x, rs):
    kurtosis_values = []
    max_components = x.shape[1]
    components_range = range(1, max_components+1)

    for n_components in components_range:
        ica = FastICA(n_components=n_components, random_state=rs)
        x_transformed = ica.fit_transform(x)
        abs_kurtosis = np.abs(kurtosis(x_transformed, axis=0))
        kurtosis_values.append(np.mean(abs_kurtosis))

    # Plot kurtosis values
    plt.plot(components_range, kurtosis_values, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Absolute Average Kurtosis')
    plt.xticks(components_range)
    plt.title('ICA Kurtosis Plot')
    plt.savefig(f'Images/{dataset}/ICA/Component Analysis.png')
    plt.close()
    return np.argmax(kurtosis_values[:max_components-1]) + 1

def perform_rp(dataset, x, rs):
    frobenius_norm = []
    mse = []
    max_components = x.shape[1]
    components_range = range(1, max_components+1)

    # Apply Random Projection and compute reconstruction error for each number of components
    for n_components in components_range:
        rp = GaussianRandomProjection(n_components=n_components, random_state=rs, compute_inverse_components=True)
        x_projected = rp.fit_transform(x)
        x_reconstructed = rp.inverse_transform(x_projected)
        reconstruction_error = np.linalg.norm(x - x_reconstructed)
        frobenius_norm.append(reconstruction_error)
        mse.append(mean_squared_error(x, x_reconstructed))

    # Plot reconstruction error
    plt.plot(components_range, frobenius_norm, marker='o', label='Frobenius Norm', color='blue')
    plt.ylabel('Frobenius Norm')
    plt.gca().twinx()
    plt.plot(components_range, mse, marker='o', label='Mean Squared Error', color='darkorange')
    plt.gca().set_ylabel('Mean Squared Error', rotation=270, labelpad=15)
    plt.xticks(components_range)
    plt.xlabel('Number of Components')
    plt.title('Random Projection Reconstruction Error')
    handles, labels = [], []
    for ax in plt.gcf().axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    plt.legend(handles, labels, loc='upper right')
    plt.savefig(f'Images/{dataset}/RP/Component Analysis.png')
    plt.close()

    # Get the optimal number of components
    return None
    
def perform_isomap(dataset, x):
    reconstruction_errors = []
    max_components = x.shape[1]
    components_range = range(1, max_components+1)

    # Loop over each number of components and evaluate the model
    for n_components in components_range:
        isomap = Isomap(n_components=n_components, n_neighbors=10)
        isomap.fit_transform(x)
        reconstruction_errors.append(isomap.reconstruction_error())

    # Plot the evaluation metric vs. the number of components
    plt.plot(components_range, reconstruction_errors, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Isomap Reconstruction Error')
    plt.xticks(components_range)
    plt.savefig(f'Images/{dataset}/Isomap/Component Analysis.png')
    plt.close()

    # Get the optimal number of components
    return None