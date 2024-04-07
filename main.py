#import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pair_confusion_matrix,
)
from sklearn.manifold import TSNE
from pyclustering.cluster.cure import cure

def read_dataset(file_path):
    # Leer el dataset desde un archivo xlsx
    X = pd.read_excel(file_path)
    return X

def normalize_data(X):
    # Normalizar los datos utilizando Min-Max
    scaler = MinMaxScaler()
    X_minmax = scaler.fit_transform(X)
    return X_minmax

def apply_clustering(X_minmax, k_range):
    metrics = {
        "Silhouette": [],
        "Calinski-Harabasz": [],
        "Davies-Bouldin": [],
    }
    models = {}
    
    for k in k_range:
        em_model = GaussianMixture(n_components=k, max_iter=1000, random_state=42, init_params='k-means++', covariance_type='spherical')
        em_labels = em_model.fit_predict(X_minmax)
        models[k] = em_model
        
        metrics["Silhouette"].append(silhouette_score(X_minmax, em_labels))
        metrics["Calinski-Harabasz"].append(calinski_harabasz_score(X_minmax, em_labels))
        metrics["Davies-Bouldin"].append(davies_bouldin_score(X_minmax, em_labels))
    
    metrics_df = pd.DataFrame(metrics, index=k_range)
    
    optimal_k = metrics_df["Silhouette"].idxmax()
    
    return models, optimal_k, metrics_df

def plot_silhouette_comparison(k_range, silhouette_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, silhouette_scores, marker="o")
    plt.title("Silhouette Score Comparison")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.show()

def plot_tsne(X_tsne):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=1)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
def plot_different_markers(X, labels,k_value):
    # Define markers you want to use
    markers = ['o', 's', '^', 'D', 'P',
                ]  # You can add more markers if needed

    # Iterate over unique labels and plot points with different markers
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], marker=markers[label], label=f'Cluster {label+1}')
    plt.title(f"t-SNE después de clustering con k={k_value}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.show()
def plot_different_colors(X, labels,k_value):
    # Define colors you want to use
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

    # Iterate over unique labels and plot points with different colors
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[label], label=f'Cluster {label+1}')
    plt.title(f"t-SNE después de clustering con k={k_value}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.show()
    
def plot_cmap(X, labels,k_value):
    # Plot t-SNE con cmap
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(f"t-SNE después de clustering con k={k_value}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar()
    plt.show()

def run_clustering_pipeline(file_path, k_range):
    X_minmax = prepare_data(file_path,remove_outliers=True)
    models, optimal_k, metrics_df = apply_clustering(X_minmax, k_range)
    
    print(metrics_df)
    print("El valor óptimo de k es:", optimal_k)
    
    silhouette_scores = metrics_df["Silhouette"]
    plot_silhouette_comparison(k_range, silhouette_scores)
    
    tsne = TSNE(n_components=2, random_state=0, metric="cosine")
    X_tsne = tsne.fit_transform(X_minmax)
    plot_tsne(X_tsne)
    
    for k_value in [optimal_k - 1, optimal_k, optimal_k + 1]:
        if k_value not in models:
            continue
        em_model = models[k_value]
        em_labels = em_model.fit_predict(X_minmax)
        plot_cmap(X_tsne, em_labels, k_value)
    

def prepare_data(file_path, remove_outliers=True):
    X = read_dataset(file_path)
    if remove_outliers:
        df_no_outiers = delete_outliers_iqr(X)
        X_minmax = normalize_data(df_no_outiers)
        return X_minmax
    else:
        X_minmax = normalize_data(X)
        return X_minmax
########
# CURE #
########

#The IQR is used to detect outliers, which is less sensitive to extreme values.
def delete_outliers_iqr(df, threshold=5):
    # Loop through each column
    for column in df.columns:
        # Skip non-numeric data
        if df[column].dtype.kind in 'bifc':
            # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the column
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            # Calculate the IQR
            IQR = Q3 - Q1
            # Define bounds for the outliers
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            
            # Filter out outliers based on index
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

def convert_clusters_to_labels(clusters):
    """
    Converts clusters from pyclustering's format to labels suitable for use with sklearn's metrics.

    :param clusters: A list of clusters where each cluster is a list of data row indices.
    :return: A list of labels, where each label corresponds to the cluster number of each row.
    """
    label_list = [0] * sum(len(cluster) for cluster in clusters)
    for cluster_label, cluster in enumerate(clusters):
        for index in cluster:
            label_list[index] = cluster_label
    return label_list

def apply_cure_clustering(df_minmax, k_range):
    metrics = {
        "Silhouette": [],
        "Calinski-Harabasz": [],
        "Davies-Bouldin": [],
    }
    labels = {}
    for k in k_range:
        cure_instance = cure(df_minmax.values.tolist(), k, 1, 0, False)
        cure_instance.process()
        clusters = cure_instance.get_clusters()
        labels_local = convert_clusters_to_labels(clusters)
        metrics["Silhouette"].append(silhouette_score(df_minmax.values, labels_local))
        metrics["Calinski-Harabasz"].append(calinski_harabasz_score(df_minmax.values, labels_local))
        metrics["Davies-Bouldin"].append(davies_bouldin_score(df_minmax.values, labels_local))
        labels[k] = labels_local

    metrics_df = pd.DataFrame(metrics, index=k_range)
    optimal_k = metrics_df["Silhouette"].idxmax()
    return labels,optimal_k, metrics_df
        
def run_cure_clustering_pipeline(file_path, k_range):
    X_minmax = prepare_data(file_path,remove_outliers=True)
    df_minmax = pd.DataFrame(X_minmax)
    labels, optimal_k, metrics_df = apply_cure_clustering(df_minmax, k_range)
    
    print(metrics_df)
    print("El valor óptimo de k es:", optimal_k)
   
    silhouette_scores = metrics_df["Silhouette"]
    plot_silhouette_comparison(k_range, silhouette_scores)
    
    tsne = TSNE(n_components=2, random_state=0, metric="cosine")
    X_tsne = tsne.fit_transform(X_minmax)
    plot_tsne(X_tsne)
    
    for k_value in [optimal_k - 1, optimal_k, optimal_k + 1]:
        if k_value not in labels:
            continue
        labels_k = labels[k_value]
        plot_cmap(X_tsne, labels_k, k_value)

if __name__ == "__main__":
    file_path = "dataset(wq).xlsx"
    k_range = range(2, 9)
    
    run_clustering_pipeline(file_path, k_range)
    run_cure_clustering_pipeline(file_path, k_range)
    # Puede probar con otros datasets y valores de k
    #file_path = "data/wholesale_customers.xlsx"
    #k_range = range(2, 11)
    #run_clustering_pipeline(file_path, k_range)