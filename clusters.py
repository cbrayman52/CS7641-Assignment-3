import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score

def gmm_bic_score(estimator, X):
    return -estimator.bic(X)

def gmm_silhouette_score(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

def gmm_evaluation(dataset, x, rs=None):

    # Define parameters for tuning
    params = {'n_components': range(2, 21),
              'covariance_type': ['spherical', 'tied', 'diag', 'full'],
              # 'max_iter': [50, 100, 500, 1000],
              # 'n_init': range(1, 12, 2),
              'init_params': ['kmeans']
              }
    
    # Perform Grid Search using BIC Score
    grid_search = GridSearchCV(GaussianMixture(random_state=rs), param_grid=params, scoring=gmm_bic_score)
    grid_search.fit(x)

    grid_search_results = pd.DataFrame(grid_search.cv_results_)[['param_n_components', 'param_covariance_type', 'param_init_params',
                                                                 # 'param_max_iter', 'param_n_init',
                                                                 'mean_test_score', 'std_test_score']]
    grid_search_results['mean_test_score'] = -grid_search_results['mean_test_score']
    grid_search_results = grid_search_results.rename(columns={'param_n_components': 'Number of Components',
                                                              'param_covariance_type': 'Type of Covariance',
                                                              # 'param_max_iter': 'Max Iterations',
                                                              # 'param_n_init': 'Number of Initializations',
                                                              'param_init_params': 'Weight Initialization Method',
                                                              'mean_test_score': 'BIC Score',
                                                              'std_test_score': 'std_dev'})
    
    # Plot the data
    grid_search_results_plot = grid_search_results[['Number of Components', 'Type of Covariance', 'BIC Score']]
    filtered_results = grid_search_results_plot[grid_search_results_plot.apply(lambda row: 'tied' in row.values, axis=1)]
    filtered_results.plot(x='Number of Components', y='BIC Score', kind='line', marker='o')
    plt.title('BIC Score vs Number of Components')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC Score')
    plt.xticks(range(0, 20), range(2, 22))
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/GMM/BIC Score.png')
    plt.close()
    
    # grid_search_results_plot = grid_search_results[['Number of Components', 'Type of Covariance', 'BIC Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Number of Components',
    #                 y='BIC Score',
    #                 hue='Type of Covariance')
    # best_score = grid_search_results_plot.loc[grid_search_results_plot['BIC Score'].idxmin()]
    # ax = g.axes[0, 0]
    # ax.axhline(y=best_score['BIC Score'], color='red', linestyle='--')
    # plt.text(-1.1, best_score['BIC Score'], f'{best_score["BIC Score"]:.2f}', 
    #      ha='center', va='center', color='red', fontsize=8)
    # plt.savefig(f'Images/{dataset}/GMM/BIC/BIC Score.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Type of Covariance', 'BIC Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Type of Covariance',
    #                 y='BIC Score')
    # plt.savefig(f'Images/{dataset}/GMM/BIC/Covariance.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Max Iterations', 'BIC Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Max Iterations',
    #                 y='BIC Score')
    # plt.savefig(f'Images/{dataset}/GMM/BIC/Max Iters.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Number of Initializations', 'BIC Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Number of Initializations',
    #                 y='BIC Score')
    # plt.savefig(f'Images/{dataset}/GMM/BIC/Num Inits.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Weight Initialization Method', 'BIC Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Weight Initialization Method',
    #                 y='BIC Score')
    # plt.savefig(f'Images/{dataset}/GMM/BIC/Init Params.png')
    # plt.close()

    # Perform Grid Search using Silhouette Score
    grid_search = GridSearchCV(GaussianMixture(random_state=rs), param_grid=params, scoring=gmm_silhouette_score)
    grid_search.fit(x)

    grid_search_results = pd.DataFrame(grid_search.cv_results_)[['param_n_components', 'param_covariance_type', 'param_init_params',
                                                                 # 'param_max_iter', 'param_n_init',
                                                                 'mean_test_score', 'std_test_score']]
    grid_search_results['mean_test_score'] = grid_search_results['mean_test_score']
    grid_search_results = grid_search_results.rename(columns={'param_n_components': 'Number of Components',
                                                              'param_covariance_type': 'Type of Covariance',
                                                              # 'param_max_iter': 'Max Iterations',
                                                              # 'param_n_init': 'Number of Initializations',
                                                              'param_init_params': 'Weight Initialization Method',
                                                              'mean_test_score': 'Silhouette Score',
                                                              'std_test_score': 'std_dev'})
    
    # Plot the data
    grid_search_results_plot = grid_search_results[['Number of Components', 'Type of Covariance', 'Silhouette Score']]
    filtered_results = grid_search_results_plot[grid_search_results_plot.apply(lambda row: 'tied' in row.values, axis=1)]
    filtered_results.plot(x='Number of Components', y='Silhouette Score', kind='line', marker='o')
    plt.title('Silhouette Score for Gaussian Mixture Model')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(0, 20), range(2, 22))
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/GMM/Silhouette Score.png')
    plt.close()

    # grid_search_results_plot = grid_search_results[['Number of Components', 'Type of Covariance', 'Silhouette Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Number of Components',
    #                 y='Silhouette Score',
    #                 hue='Type of Covariance')
    # best_score = grid_search_results_plot.loc[grid_search_results_plot['Silhouette Score'].idxmax()]
    # ax = g.axes[0, 0]
    # ax.axhline(y=best_score['Silhouette Score'], color='red', linestyle='--')
    # plt.text(-1, best_score['Silhouette Score'], f'{best_score["Silhouette Score"]:.2f}', 
    #      ha='center', va='center', color='red', fontsize=8)
    # plt.savefig(f'Images/{dataset}/GMM/Silhouette/Silhouette Score.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Type of Covariance', 'Silhouette Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Type of Covariance',
    #                 y='Silhouette Score')
    # plt.savefig(f'Images/{dataset}/GMM/Silhouette/Covariance.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Max Iterations', 'Silhouette Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Max Iterations',
    #                 y='Silhouette Score')
    # plt.savefig(f'Images/{dataset}/GMM/Silhouette/Max Iters.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Number of Initializations', 'Silhouette Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Number of Initializations',
    #                 y='Silhouette Score')
    # plt.savefig(f'Images/{dataset}/GMM/Silhouette/Num Inits.png')
    # plt.close()

    # grid_search_results_plot = grid_search_results[['Weight Initialization Method', 'Silhouette Score']]
    # g = sns.catplot(data=grid_search_results_plot,
    #                 kind='bar',
    #                 x='Weight Initialization Method',
    #                 y='Silhouette Score')
    # plt.savefig(f'Images/{dataset}/GMM/Silhouette/Init Params.png')
    # plt.close()
    
    return


def kmeans_evaluation(dataset, x, rs=None):
    # Calculate WCSS for different values of k
    wcss = []
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=rs)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(1, 21), wcss, marker='o', linestyle='-')
    plt.title('Scree Plot')
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
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=rs)
        cluster_labels = kmeans.fit_predict(x)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(x, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot silhouette scores for different numbers of clusters
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score for K-Means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/K Means/Silhouette Score.png')
    plt.close()

    return

def predict_labels(model, x, true_labels):

    model.fit(x)
    predicted_labels = model.predict(x)

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    return ari, nmi