import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "4"

from datasets import get_datasets
import gaussian_mixture
import k_means
import dim_red

rs = 52

dataset1, dataset2 = get_datasets(rs)

#########################################################################################################
# 1. Apply the clustering algorithms on the datasets. You will report on each of the clustering         #
#    algorithms for each dataset, resulting in 4 demonstrations.                                        #
#########################################################################################################

# Guassian Mixture Model
# gaussian_mixture.cluster_performance_evaluation('Wine Quality', dataset1['x1_train_scaled'], rs)
# gaussian_mixture.cluster_performance_evaluation('Generated Dataset', dataset2['x2_train_scaled'], rs)

# K Means
# k_means.cluster_performance_evaluation('Wine Quality', dataset1['x1_train_scaled'], rs)
# k_means.cluster_performance_evaluation('Generated Dataset', dataset2['x2_train_scaled'], rs)

#########################################################################################################
# 2. Apply the dimensionality reduction algorithms on the datasets. You will report on each of          #
#    the dimensionality reduction algorithms, resulting in 8 demonstrations.                            #
#########################################################################################################

# PCA
num_components = dim_red.perform_pca('Wine Quality', dataset1['x1_train_scaled'], rs)
num_components = dim_red.perform_pca('Generated Dataset', dataset2['x2_train_scaled'], rs)

# ICA
num_components = dim_red.perform_ica('Wine Quality', dataset1['x1_train_scaled'], rs)
num_components = dim_red.perform_ica('Generated Dataset', dataset2['x2_train_scaled'], rs)

# Randomized Projections
num_components = dim_red.perform_rp('Wine Quality', dataset1['x1_train_scaled'], rs)
num_components = dim_red.perform_rp('Generated Dataset', dataset2['x2_train_scaled'], rs)

# Isomap
num_components = dim_red.perform_isomap('Wine Quality', dataset1['x1_train_scaled'])
num_components = dim_red.perform_isomap('Generated Dataset', dataset2['x2_train_scaled'])

#########################################################################################################
# 3. Re-apply the clustering algorithms on the set of dimensionality reduction datasets. This will      #
#    result in 16 combinations of results of datasets, dimensionality reduction, and clustering methods.#
#########################################################################################################

# PCA - Guassian Mixture Model - Dataset 1
# pca = PCA(n_components=8, random_state=rs)
# pca.fit(dataset1['x1_train_scaled'])
# gaussian_mixture.cluster_performance_evaluation('Wine Quality', dataset1['x1_train_scaled'], rs)
# PCA - K Means - Dataset 1

# ICA - Guassian Mixture Model - Dataset 1

# ICA - K Means - Dataset 1

# Randomized Projections - Guassian Mixture Model - Dataset 1

# Randomized Projections - K Means - Dataset 1

# Isomap - Guassian Mixture Model - Dataset 1

# Isomap - K Means - Dataset 1


#########################################################################################################
# 4. Choose one of your datasets. Re-run your neural network learner from Assignment #1 with each of    #
#    the dimensionality reduction algorithms applied. You will report on a different linear method from #
#    Step 3 (PCA, ICA, or RP) and on the manifold learning algorithm of your choice, resulting in 2     #
#    total demonstrations.                                                                              #
#########################################################################################################


#########################################################################################################
# 5. Using the same dataset as Step 4, use both previously generated clusters from Step 1 as new        #
#    features in your dataset. Again, rerun your neural network learner on the newly projected data and #
#    note the findings. You will report on each of the clustering algorithms, resulting in 2            #
#    demonstrations.                                                                                    #
#########################################################################################################

