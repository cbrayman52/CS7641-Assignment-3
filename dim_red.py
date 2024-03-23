import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def perform_pca(dataset, x, rs=None):

    pca = PCA(n_components=x.shape[1], random_state=rs)
    pca.fit_transform(x)
    eigenvalues = pca.explained_variance_

    # Plot eigenvalues
    plt.plot(range(1, pca.n_components_ + 1), eigenvalues, marker='o', label='Eigenvalues')
    plt.xlabel('Components')
    plt.ylabel('Eigenvalues')
    plt.title('PCA Scree Plot')
    for z, y in zip(range(1, pca.n_components_ + 1), eigenvalues):
        plt.annotate(f'{y:.2f}', (z, y), textcoords='offset points', xytext=(5,5), ha='center', weight='bold', color='blue')

    # Plot cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, pca.n_components_ + 1), cumulative_variance, marker='o', label='Cumulative Explained Variance')
    for z, y in zip(range(1, pca.n_components_ + 1), cumulative_variance):
        plt.annotate(f'{y:.2f}', (z, y), textcoords='offset points', xytext=(-5,5), ha='center', weight='bold', color='darkorange')

    plt.xticks(range(1, pca.n_components_ + 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Images/{dataset}/PCA Component Analysis.png')
    plt.close()

    return np.where(cumulative_variance >= 0.8)[0][0]

def perform_ica(dataset, x, rs=None):

    ica = FastICA(n_components=x.shape[1], random_state=rs)
    x_transformed = ica.fit_transform(x)
    kurtosis_values = np.abs(kurtosis(x_transformed))

    sorted_indices = np.argsort(kurtosis_values)[::-1]
    sorted_values = kurtosis_values[sorted_indices]
    sorted_indices += 1

    # Plot kurtosis values
    plt.plot(range(len(sorted_values)), sorted_values, marker='o')
    plt.xlabel('Component')
    plt.ylabel('Kurtosis')
    plt.xticks(range(len(sorted_values)), range(1, x.shape[1]+1))
    plt.title('ICA Kurtosis Plot')
    for z, y in enumerate(sorted_values):
        plt.annotate(f'{y:.2f}', (z, y), textcoords='offset points', xytext=(10,5), ha='center', weight='bold', color='blue')
    plt.savefig(f'Images/{dataset}/ICA Component Analysis.png')
    plt.close()

    # Get the optimal number of components
    return None

def perform_rp(dataset, x, rs=None):

    # Initialize lists to store results across iterations
    frobenius_norm_all = []
    mse_all = []
    max_components = x.shape[1]
    components_range = range(1, max_components+1)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Loop over five iterations
    for i in range(5):

        rs = np.random.randint(1000)
        frobenius_norm = []
        mse = []

        # Apply Random Projection and compute reconstruction error for each number of components
        for n_components in components_range:
            rp = GaussianRandomProjection(n_components=n_components, random_state=rs, compute_inverse_components=True)
            x_projected = rp.fit_transform(x_scaled)
            x_reconstructed = rp.inverse_transform(x_projected)
            x_reconstructed = scaler.inverse_transform(x_reconstructed)
            reconstruction_error = np.linalg.norm(x - x_reconstructed)
            frobenius_norm.append(reconstruction_error)
            mse.append(mean_squared_error(x, x_reconstructed))

        frobenius_norm_all.append(frobenius_norm)
        mse_all.append(mse)

    # Calculate the mean and standard deviation across all iterations
    mean_frobenius_norm = np.mean(frobenius_norm_all, axis=0)
    std_frobenius_norm = np.std(frobenius_norm_all, axis=0)
    mean_mse = np.mean(mse_all, axis=0)
    std_mse = np.std(mse_all, axis=0)

    # Plot reconstruction error
    plt.plot(components_range, mean_frobenius_norm, marker='o', label='Frobenius Norm')
    plt.fill_between(components_range, mean_frobenius_norm - std_frobenius_norm, mean_frobenius_norm + std_frobenius_norm, color='blue', alpha=0.2)
    for z, y in zip(range(1, rp.n_components_ + 1), mean_frobenius_norm):
        plt.annotate(f'{y:.2f}', (z, y), textcoords='offset points', xytext=(5,5), ha='center', weight='bold', color='blue')
    plt.ylabel('Frobenius Norm')
    plt.xlabel('Components')

    plt.gca().twinx()
    plt.plot(components_range, mean_mse, marker='o', label='Mean Squared Error', color='orange')
    plt.fill_between(components_range, mean_mse - std_mse, mean_mse + std_mse, color='orange', alpha=0.2)
    for z, y in zip(range(1, rp.n_components_ + 1), mean_mse):
        plt.annotate(f'{y:.2f}', (z, y), textcoords='offset points', xytext=(0,-15), ha='center', weight='bold', color='darkorange')
    plt.gca().set_ylabel('Mean Squared Error', rotation=270, labelpad=15)
    plt.xticks(components_range)
    plt.xlabel('Components')
    plt.title('Random Projection Reconstruction Error')
    
    handles, labels = [], []
    for ax in plt.gcf().axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    plt.legend(handles, labels, loc='upper right')
    plt.savefig(f'Images/{dataset}/RP Component Analysis.png')
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
    plt.xlabel('Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Isomap Reconstruction Error')
    plt.xticks(components_range)
    for z, y in zip(range(1, isomap.embedding_.shape[1] + 1), reconstruction_errors):
        plt.annotate(f'{y:.2f}', (z, y), textcoords='offset points', xytext=(5,5), ha='center', weight='bold', color='blue')
    plt.savefig(f'Images/{dataset}/Isomap Component Analysis.png')
    plt.close()

    # Get the optimal number of components
    return None