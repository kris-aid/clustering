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
    X = read_dataset(file_path)
    X_minmax = normalize_data(X)
    models, optimal_k, metrics_df = apply_clustering(X_minmax, k_range)
    
    print(metrics_df)
    print("El valor óptimo de k es:", optimal_k)
    
    silhouette_scores = metrics_df["Silhouette"]
    plot_silhouette_comparison(k_range, silhouette_scores)
    
    tsne = TSNE(n_components=2, random_state=0, metric="cosine")
    X_tsne = tsne.fit_transform(X_minmax)
    plot_tsne(X_tsne)
    
    for k_value in [optimal_k - 1, optimal_k, optimal_k + 1]:
        em_model = models[k_value]
        em_labels = em_model.fit_predict(X_minmax)
        plot_cmap(X_tsne, em_labels, k_value)
    
if __name__ == "__main__":
    file_path = "dataset(wq).xlsx"
    k_range = range(2, 9)
    
    run_clustering_pipeline(file_path, k_range)
    
    # Puede probar con otros datasets y valores de k
    #file_path = "data/wholesale_customers.xlsx"
    #k_range = range(2, 11)
    #run_clustering_pipeline(file_path, k_range)