import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier

import os
os.environ["OMP_NUM_THREADS"] = "4"

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

# Guassian Mixture Model
clusters.gmm_evaluation('Step 1/Wine Quality', dataset1['x1_train_scaled'], rs)
clusters.gmm_evaluation('Step 1/Generated Dataset', dataset2['x2_train_scaled'], rs)

# K Means
clusters.kmeans_evaluation('Step 1/Wine Quality', dataset1['x1_train_scaled'], rs)
clusters.kmeans_evaluation('Step 1/Generated Dataset', dataset2['x2_train_scaled'], rs)

#########################################################################################################
# 2. Apply the dimensionality reduction algorithms on the datasets. You will report on each of          #
#    the dimensionality reduction algorithms, resulting in 8 demonstrations.                            #
#########################################################################################################

# PCA
num_components = dim_red.perform_pca('Step 2/Wine Quality', dataset1['x1_train_scaled'], rs)
num_components = dim_red.perform_pca('Step 2/Generated Dataset', dataset2['x2_train_scaled'], rs)

# ICA
num_components = dim_red.perform_ica('Step 2/Wine Quality', dataset1['x1_train_scaled'], rs)
num_components = dim_red.perform_ica('Step 2/Generated Dataset', dataset2['x2_train_scaled'], rs)

# Randomized Projections
num_components = dim_red.perform_rp('Step 2/Wine Quality', dataset1['x1_train_scaled'], rs)
num_components = dim_red.perform_rp('Step 2/Generated Dataset', dataset2['x2_train_scaled'], rs)

# Isomap
num_components = dim_red.perform_isomap('Step 2/Wine Quality', dataset1['x1_train_scaled'])
num_components = dim_red.perform_isomap('Step 2/Generated Dataset', dataset2['x2_train_scaled'])

#########################################################################################################
# 3. Re-apply the clustering algorithms on the set of dimensionality reduction datasets. This will      #
#    result in 16 combinations of results of datasets, dimensionality reduction, and clustering methods.#
#########################################################################################################

# PCA - Guassian Mixture Model - Dataset 1
pca = PCA(n_components=8, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality', x_reduced, rs)

# PCA - Guassian Mixture Model - Dataset 2
pca = PCA(n_components=8, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# PCA - K Means - Dataset 1
pca = PCA(n_components=8, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality', x_reduced, rs)

# PCA - K Means - Dataset 2
pca = PCA(n_components=8, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# ICA - Guassian Mixture Model - Dataset 1
ica = FastICA(n_components=8, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality', x_reduced, rs)

# ICA - Guassian Mixture Model - Dataset 2
ica = FastICA(n_components=8, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# ICA - K Means - Dataset 1
ica = FastICA(n_components=8, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality', x_reduced, rs)

# ICA - K Means - Dataset 2
ica = FastICA(n_components=8, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# Randomized Projections - Guassian Mixture Model - Dataset 1
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality', x_reduced, rs)

# Randomized Projections - Guassian Mixture Model - Dataset 2
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# Randomized Projections - K Means - Dataset 1
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x1_train_scaled'])
clusters.kmeans_evaluation('Step 3/Wine Quality', x_reduced, rs)

# Randomized Projections - K Means - Dataset 2
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x2_train_scaled'])
clusters.kmeans_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# Isomap - Guassian Mixture Model - Dataset 1
isomap = Isomap(n_components=8)
x_reduced = isomap.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality', x_reduced, rs)

# Isomap - Guassian Mixture Model - Dataset 2
isomap = Isomap(n_components=8)
x_reduced = isomap.fit_transform(dataset2['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset', x_reduced, rs)

# Isomap - K Means - Dataset 1
isomap = Isomap(n_components=8)
x_reduced = isomap.fit_transform(dataset1['x1_train_scaled'])
clusters.gmm_evaluation('Step 3/Wine Quality', x_reduced, rs)

# Isomap - K Means - Dataset 2
isomap = Isomap(n_components=8)
x_reduced = isomap.fit_transform(dataset2['x2_train_scaled'])
clusters.gmm_evaluation('Step 3/Generated Dataset', x_reduced, rs)

#########################################################################################################
# 4. Choose one of your datasets. Re-run your neural network learner from Assignment #1 with each of    #
#    the dimensionality reduction algorithms applied. You will report on a different linear method from #
#    Step 3 (PCA, ICA, or RP) and on the manifold learning algorithm of your choice, resulting in 2     #
#    total demonstrations.                                                                              #
#########################################################################################################

# PCA - Dataset 1
pca = PCA(n_components=8, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x1_train_scaled'])
nn.analyze_nn('Step 4/Wine Quality', x_reduced, rs)

# PCA - Dataset 2
pca = PCA(n_components=8, random_state=rs)
x_reduced = pca.fit_transform(dataset1['x2_train_scaled'])
nn.analyze_nn('Step 4/Generated Dataset', x_reduced, rs)

# ICA - Dataset 1
ica = FastICA(n_components=8, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x1_train_scaled'])
nn.analyze_nn('Step 4/Wine Quality', x_reduced, rs)

# ICA - Dataset 2
ica = FastICA(n_components=8, random_state=rs)
x_reduced = ica.fit_transform(dataset1['x2_train_scaled'])
nn.analyze_nn('Step 4/Generated Dataset', x_reduced, rs)

# Randomized Projections - Dataset 1
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x1_train_scaled'])
nn.analyze_nn('Step 4/Wine Quality', x_reduced, rs)

# Randomized Projections - Dataset 2
rp = GaussianRandomProjection(n_components=8, random_state=rs)
x_reduced = rp.fit_transform(dataset1['x2_train_scaled'])
nn.analyze_nn('Step 4/Generated Dataset', x_reduced, rs)

# Isomap - Dataset 1
isomap = Isomap(n_components=8)
x_reduced = isomap.fit_transform(dataset1['x1_train_scaled'])
nn.analyze_nn('Step 4/Wine Quality', x_reduced, rs)

# Isomap - Dataset 2
isomap = Isomap(n_components=8)
x_reduced = isomap.fit_transform(dataset1['x2_train_scaled'])
nn.analyze_nn('Step 4/Generated Dataset', x_reduced, rs)

#########################################################################################################
# 5. Using the same dataset as Step 4, use both previously generated clusters from Step 1 as new        #
#    features in your dataset. Again, rerun your neural network learner on the newly projected data and #
#    note the findings. You will report on each of the clustering algorithms, resulting in 2            #
#    demonstrations.                                                                                    #
#########################################################################################################

