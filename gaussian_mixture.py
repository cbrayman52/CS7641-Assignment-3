import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "4"

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import silhouette_score

def gmm_bic_score(estimator, X):
    return -estimator.bic(X)

def gmm_silhouette_score(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

def cluster_performance_evaluation(dataset, x, rs):

    # Define parameters for tuning
    params = {'n_components': range(2, 11),
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
    
    grid_search_results_plot = grid_search_results[['Number of Components', 'Type of Covariance', 'BIC Score']]
    g = sns.catplot(data=grid_search_results_plot,
                    kind='bar',
                    x='Number of Components',
                    y='BIC Score',
                    hue='Type of Covariance')
    best_score = grid_search_results_plot.loc[grid_search_results_plot['BIC Score'].idxmin()]
    ax = g.axes[0, 0]
    ax.axhline(y=best_score['BIC Score'], color='red', linestyle='--')
    plt.text(-1.1, best_score['BIC Score'], f'{best_score["BIC Score"]:.2f}', 
         ha='center', va='center', color='red', fontsize=8)
    plt.savefig(f'Images/{dataset}/GMM/BIC/BIC Score.png')
    plt.close()

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
    
    grid_search_results_plot = grid_search_results[['Number of Components', 'Type of Covariance', 'Silhouette Score']]
    g = sns.catplot(data=grid_search_results_plot,
                    kind='bar',
                    x='Number of Components',
                    y='Silhouette Score',
                    hue='Type of Covariance')
    best_score = grid_search_results_plot.loc[grid_search_results_plot['Silhouette Score'].idxmax()]
    ax = g.axes[0, 0]
    ax.axhline(y=best_score['Silhouette Score'], color='red', linestyle='--')
    plt.text(-1, best_score['Silhouette Score'], f'{best_score["Silhouette Score"]:.2f}', 
         ha='center', va='center', color='red', fontsize=8)
    plt.savefig(f'Images/{dataset}/GMM/Silhouette/Silhouette Score.png')
    plt.close()

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

    
    # Define covariance types
    covariance_types = ['spherical', 'tied', 'diag', 'full']

    # Plot all covariance types on the same plot
    plt.figure(figsize=(10, 8))

    for cov_type in covariance_types:
        bic = []
        for k in range(1, 21):
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type)
            gmm.fit(x)
            bic.append(gmm.bic(x))
            
        # Plot the elbow curve for the current covariance type
        plt.plot(range(1, 21), bic, label=cov_type)

    plt.title('Elbow Method for Gaussian Mixture Model')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC Score')
    plt.xticks(np.arange(1, 21, 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/GMM/BIC Score.png')
    plt.close()

    for cov_type in covariance_types:
        silhouette_scores = []
        for k in range(2, 21):
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
            cluster_labels = gmm.fit_predict(x)
            silhouette_avg = silhouette_score(x, cluster_labels)
            silhouette_scores.append(silhouette_avg)
                
        # Plot the silhouette score curve for the current covariance type
        plt.plot(range(2, 21), silhouette_scores, label=cov_type)

    plt.title('Silhouette Score for Gaussian Mixture Model')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(np.arange(1, 21, 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Images/{dataset}/GMM/Silhouette Score.png')
    plt.close()

    return