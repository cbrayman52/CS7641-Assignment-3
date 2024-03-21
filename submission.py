import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import datasets
import clusters
import dim_red
import nn

rs = 52

dataset1, dataset2 = datasets.get_datasets(rs)

#########################################################################################################
# 1. Apply the clustering algorithms on the datasets. You will report on each of the clustering         #
#    algorithms for each dataset, resulting in 4 demonstrations.                                        #
#########################################################################################################
'''
ari = [0] * 4
nmi = [0] * 4
# Guassian Mixture Model - Dataset 1
clusters.gmm_evaluation('Step 1/Wine Quality', dataset1['x1_train_scaled'], rs)
ari[0], nmi[0] = clusters.predict_labels(GaussianMixture(n_components=2, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Guassian Mixture Model - Dataset 2
clusters.gmm_evaluation('Step 1/Generated Dataset', dataset2['x2_train_scaled'], rs)
ari[1], nmi[1] = clusters.predict_labels(GaussianMixture(n_components=11, covariance_type='full', n_init=5, random_state=rs), dataset2['x2_train_scaled'], dataset2['y2_train'])

# K Means - Dataset 1
clusters.kmeans_evaluation('Step 1/Wine Quality', dataset1['x1_train_scaled'], rs)
ari[2], nmi[2] = clusters.predict_labels(KMeans(n_clusters=8, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# K Means - Dataset 2
clusters.kmeans_evaluation('Step 1/Generated Dataset', dataset2['x2_train_scaled'], rs)
ari[3], nmi[3] = clusters.predict_labels(KMeans(n_clusters=11, n_init=10, random_state=rs), dataset2['x2_train_scaled'], dataset2['y2_train'])

#########################################################################################################
# 2. Apply the dimensionality reduction algorithms on the datasets. You will report on each of          #
#    the dimensionality reduction algorithms, resulting in 8 demonstrations.                            #
#########################################################################################################

# PCA - Dataset 1
num_components = dim_red.perform_pca('Step 2/Wine Quality', dataset1['x1_train_scaled'], rs)

# PCA - Dataset 2
num_components = dim_red.perform_pca('Step 2/Generated Dataset', dataset2['x2_train_scaled'], rs)

# ICA - Dataset 1
num_components = dim_red.perform_ica('Step 2/Wine Quality', dataset1['x1_train_scaled'], rs)

# ICA - Dataset 2
num_components = dim_red.perform_ica('Step 2/Generated Dataset', dataset2['x2_train_scaled'], rs)

# Randomized Projections - Dataset 1
num_components = dim_red.perform_rp('Step 2/Wine Quality', dataset1['x1_train'], rs)

# Randomized Projections - Dataset 2
num_components = dim_red.perform_rp('Step 2/Generated Dataset', dataset2['x2_train'], rs)

# Isomap - Dataset 1
num_components = dim_red.perform_isomap('Step 2/Wine Quality', dataset1['x1_train_scaled'])

# Isomap - Dataset 2
num_components = dim_red.perform_isomap('Step 2/Generated Dataset', dataset2['x2_train_scaled'])
'''
#########################################################################################################
# 3. Re-apply the clustering algorithms on the set of dimensionality reduction datasets. This will      #
#    result in 16 combinations of results of datasets, dimensionality reduction, and clustering methods.#
#########################################################################################################

ari = [0] * 16
nmi = [0] * 16

# PCA - Guassian Mixture Model - Dataset 1
pca = PCA(n_components=5, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality/PCA', x_reduced, rs)
ari[0], nmi[0] = clusters.predict_labels(GaussianMixture(n_components=5, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# PCA - Guassian Mixture Model - Dataset 2
pca = PCA(n_components=4, random_state=rs)
x_reduced = pca.fit_transform(dataset2['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset/PCA', x_reduced, rs)
ari[1], nmi[1] = clusters.predict_labels(GaussianMixture(n_components=4, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# PCA - K Means - Dataset 1
pca = PCA(n_components=5, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality/PCA', x_reduced, rs)
ari[2], nmi[2] = clusters.predict_labels(KMeans(n_clusters=5, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# PCA - K Means - Dataset 2
pca = PCA(n_components=4, random_state=rs)
x_reduced = pca.fit_transform(dataset2['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset/PCA', x_reduced, rs)
ari[3], nmi[3] = clusters.predict_labels(KMeans(n_clusters=4, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# ICA - Guassian Mixture Model - Dataset 1
ica = FastICA(n_components=5, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality/ICA', x_reduced, rs)
ari[4], nmi[4] = clusters.predict_labels(GaussianMixture(n_components=5, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# ICA - Guassian Mixture Model - Dataset 2
ica = FastICA(n_components=3, random_state=rs)
x_reduced = ica.fit_transform(dataset2['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset/ICA', x_reduced, rs)
ari[5], nmi[5] = clusters.predict_labels(GaussianMixture(n_components=3, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# ICA - K Means - Dataset 1
ica = FastICA(n_components=5, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality/ICA', x_reduced, rs)
ari[6], nmi[6] = clusters.predict_labels(KMeans(n_clusters=5, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# ICA - K Means - Dataset 2
ica = FastICA(n_components=3, random_state=rs)
x_reduced = ica.fit_transform(dataset2['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset/ICA', x_reduced, rs)
ari[7], nmi[7] = clusters.predict_labels(KMeans(n_clusters=3, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Randomized Projections - Guassian Mixture Model - Dataset 1
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality/RP', x_reduced, rs)
ari[8], nmi[8] = clusters.predict_labels(GaussianMixture(n_components=8, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Randomized Projections - Guassian Mixture Model - Dataset 2
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset2['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset/RP', x_reduced, rs)
ari[9], nmi[9] = clusters.predict_labels(GaussianMixture(n_components=8, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Randomized Projections - K Means - Dataset 1
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality/RP', x_reduced, rs)
ari[10], nmi[10] = clusters.predict_labels(KMeans(n_clusters=8, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Randomized Projections - K Means - Dataset 2
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset2['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset/RP', x_reduced, rs)
ari[11], nmi[11] = clusters.predict_labels(KMeans(n_clusters=8, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Isomap - Guassian Mixture Model - Dataset 1
isomap = Isomap(n_components=4, n_neighbors=10)
x_reduced = isomap.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality/Isomap', x_reduced, rs)
ari[12], nmi[12] = clusters.predict_labels(GaussianMixture(n_components=4, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Isomap - Guassian Mixture Model - Dataset 2
isomap = Isomap(n_components=4, n_neighbors=10)
x_reduced = isomap.fit_transform(dataset2['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset/Isomap', x_reduced, rs)
ari[13], nmi[13] = clusters.predict_labels(GaussianMixture(n_components=4, covariance_type='tied', n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Isomap - K Means - Dataset 1
isomap = Isomap(n_components=4, n_neighbors=10)
x_reduced = isomap.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality/Isomap', x_reduced, rs)
ari[14], nmi[14] = clusters.predict_labels(KMeans(n_clusters=4, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

# Isomap - K Means - Dataset 2
isomap = Isomap(n_components=4, n_neighbors=10)
x_reduced = isomap.fit_transform(dataset2['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset/Isomap', x_reduced, rs)
ari[15], nmi[15] = clusters.predict_labels(KMeans(n_clusters=4, n_init=5, random_state=rs), dataset1['x1_train_scaled'], dataset1['y1_train'])

#########################################################################################################
# 4. Choose one of your datasets. Re-run your neural network learner from Assignment #1 with each of    #
#    the dimensionality reduction algorithms applied. You will report on a different linear method from #
#    Step 3 (PCA, ICA, or RP) and on the manifold learning algorithm of your choice, resulting in 2     #
#    total demonstrations.                                                                              #
#########################################################################################################

# Baseline - Dataset 1
nn.learning_curves('Step 4/Wine Quality/Baseline', dataset1['x1_train_scaled'], dataset1['x1_test_scaled'], dataset1['y1_train'], dataset1['y1_test'], rs)
nn.validation_curves('Step 4/Wine Quality/Baseline', dataset1['x1_train_scaled'], dataset1['y1_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality/Baseline', dataset1['x1_train_scaled'], dataset1['y1_train'], rs)

# Baseline - Dataset 2
nn.learning_curves('Step 4/Generated Dataset/Baseline', dataset2['x2_train_scaled'], dataset2['x2_test_scaled'], dataset2['y2_train'], dataset2['y2_test'], rs)
nn.validation_curves('Step 4/Generated Dataset/Baseline', dataset2['x2_train_scaled'], dataset2['y2_train'], rs)
nn.accuracy_curves('Step 4/Generated Dataset/Baseline', dataset2['x2_train_scaled'], dataset2['y2_train'], rs)

# PCA - Dataset 1
pca = PCA(n_components=4, random_state=rs)
x_train_reduced = pca.fit_transform(dataset1['x1_train_scaled'])
x_test_reduced = pca.transform(dataset1['x1_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset1['y1_train'], dataset1['y1_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)

# PCA - Dataset 2
pca = PCA(n_components=4, random_state=rs)
x_train_reduced = pca.fit_transform(dataset2['x2_train_scaled'])
x_test_reduced = pca.transform(dataset2['x2_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset2['y2_train'], dataset2['y2_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)

# ICA - Dataset 1
ica = FastICA(n_components=5, random_state=rs)
x_train_reduced = ica.fit_transform(dataset1['x1_train_scaled'])
x_test_reduced = ica.transform(dataset1['x1_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset1['y1_train'], dataset1['y1_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)

# ICA - Dataset 2
ica = FastICA(n_components=3, random_state=rs)
x_train_reduced = ica.fit_transform(dataset2['x2_train_scaled'])
x_test_reduced = ica.transform(dataset2['x2_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset2['y2_train'], dataset2['y2_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)

# Randomized Projections - Dataset 1
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_train_reduced = rp.fit_transform(dataset1['x1_train_scaled'])
x_test_reduced = rp.transform(dataset1['x1_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset1['y1_train'], dataset1['y1_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)

# Randomized Projections - Dataset 2
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_train_reduced = rp.fit_transform(dataset2['x2_train_scaled'])
x_test_reduced = rp.transform(dataset2['x2_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset2['y2_train'], dataset2['y2_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)

# Isomap - Dataset 1
isomap = Isomap(n_components=4)
x_train_reduced = isomap.fit_transform(dataset1['x1_train_scaled'])
x_test_reduced = isomap.transform(dataset1['x1_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset1['y1_train'], dataset1['y1_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset1['y1_train'], rs)

# Isomap - Dataset 2
isomap = Isomap(n_components=4)
x_train_reduced = isomap.fit_transform(dataset2['x2_train_scaled'])
x_test_reduced = isomap.transform(dataset2['x2_test_scaled'])
nn.learning_curves('Step 4/Wine Quality', x_train_reduced, x_test_reduced, dataset2['y2_train'], dataset2['y2_test'], rs)
nn.validation_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)
nn.accuracy_curves('Step 4/Wine Quality', x_train_reduced, dataset2['y2_train'], rs)

#########################################################################################################
# 5. Using the same dataset as Step 4, use both previously generated clusters from Step 1 as new        #
#    features in your dataset. Again, rerun your neural network learner on the newly projected data and #
#    note the findings. You will report on each of the clustering algorithms, resulting in 2            #
#    demonstrations.                                                                                    #
#########################################################################################################

# GMM - Dataset 1
gmm = GaussianMixture(n_components=2, random_state=rs)
cluster_assignments = gmm.fit(dataset1['x1_train_scaled'])

# GMM - Dataset 2
gmm = GaussianMixture(n_components=5, random_state=rs)
cluster_assignments = gmm.fit(dataset2['x2_train_scaled'])

# KMeans - Dataset 1
kmeans = KMeans(n_clusters=2, random_state=rs)
cluster_assignments = kmeans.fit(dataset1['x1_train_scaled'])

# KMeans - Dataset 2
kmeans = KMeans(n_clusters=5, random_state=rs)
cluster_assignments = kmeans.fit(dataset2['x2_train_scaled'])