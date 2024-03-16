import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "4"

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_performance_evaluation(dataset, x, rs):

    # Calculate WCSS for different values of k
    wcss = []
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=rs)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method for K-Means Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(np.arange(1, 21, 1))
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/K Means/Elbow Score.png')
    plt.close()

    # Perform K-Means clustering for each cluster number and compute silhouette scores
    silhouette_scores = []
    cluster_range = range(2, 21)

    for n_clusters in cluster_range:
        # Fit K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(x)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(x, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot silhouette scores for different numbers of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/K Means/Silhouette Score.png')
    plt.close()

    return