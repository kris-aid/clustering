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

# Leer el dataset desde un archivo xlsx
X = pd.read_excel("dataset(wq).xlsx")

#print(X.head())

# Normalizar los datos utilizando Min-Max
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)
#print(X_minmax)

# Definir rangos de k
k_range = range(2, 9)

# Inicializar listas para almacenar métricas
metrics = {
    # "ARI": [],
    # "AMI": [],
    # "Homogeneity": [],
    # "Completeness": [],
    # "V-measure": [],
    # "FM": [],
    #  "Pair Confusion Matrix": [], # Agregar métrica adicional para comparar con el método de clustering jerárquico
    "Silhouette": [],
    "Calinski-Harabasz": [],
    "Davies-Bouldin": [],
   
}

# Aplicar algoritmos de clustering y calcular métricas para cada valor de k
for k in k_range:
    # Aplicar CURE
    # Implementación de CURE no está disponible en scikit-learn, puedes usar otra librería o implementarlo tú mismo
    print(f"Aplicando CURE con k={k}...")
    # Aplicar Expectation-Maximization
    em_model = GaussianMixture(n_components=k)
    em_labels = em_model.fit_predict(X_minmax)

    # Calcular métricas
    #Estas dependen de los labels verdaderos, por lo que no se pueden calcular en este caso
    # metrics["ARI"].append(adjusted_rand_score(y, em_labels))
    # metrics["AMI"].append(adjusted_mutual_info_score(y, em_labels))
    # metrics["Homogeneity"].append(homogeneity_score(y, em_labels))
    # metrics["Completeness"].append(completeness_score(y, em_labels))
    # metrics["V-measure"].append(v_measure_score(y, em_labels))
    # metrics["FM"].append(fowlkes_mallows_score(y, em_labels))
    #metrics["Pair Confusion Matrix"].append(pair_confusion_matrix(y, em_labels))
    
    metrics["Silhouette"].append(silhouette_score(X_minmax, em_labels))
    metrics["Calinski-Harabasz"].append(calinski_harabasz_score(X_minmax, em_labels))
    metrics["Davies-Bouldin"].append(davies_bouldin_score(X_minmax, em_labels))
    
# Convertir a DataFrame para visualización
metrics_df = pd.DataFrame(metrics, index=k_range)

# print de silueta
print(metrics_df)

# Encontrar el k óptimo basado en alguna métrica (por ejemplo, Silhouette)
optimal_k = 4 #metrics_df["Silhouette"].idxmax()

# Imprimir el valor de k óptimo
print("El valor óptimo de k es:", optimal_k)

#Plot the silhouette score comparison
plt.figure(figsize=(8, 6))
plt.plot(k_range, metrics_df["Silhouette"], marker="o")
plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(k_range)
plt.show()


# Plot t-SNE para el espacio original
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_minmax)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=em_labels, cmap='viridis')
plt.title("t-SNE después de clustering con k óptimo")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Etiqueta de cluster")
plt.show()

# Plot t-SNE después de aplicado el método de clustering
for k_value in [optimal_k - 1, optimal_k, optimal_k + 1]:
    # Aplicar clustering con el valor de k seleccionado
    em_model = GaussianMixture(n_components=k_value)
    em_labels = em_model.fit_predict(X_minmax)
    
    # Plot t-SNE
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=em_labels, cmap='viridis')
    plt.title(f"t-SNE después de clustering con k={k_value}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar(label="Etiqueta de cluster")
    plt.show()
