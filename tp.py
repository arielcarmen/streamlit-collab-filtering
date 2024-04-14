import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_id = np.argmin(distances)
        clusters[cluster_id].append(point)
    return clusters

def calculate_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)
    return centroids

def k_means(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = None
    while not np.array_equal(centroids, prev_centroids):
        prev_centroids = centroids
        clusters = assign_clusters(data, centroids)
        centroids = calculate_centroids(clusters)
    return centroids, clusters

def display_results(centroids, clusters, data):
    st.write("Centroïdes:")
    for i, centroid in enumerate(centroids):
        st.write(f"Centroïde {i+1}: {centroid}")
        cluster = clusters[i]
        st.write(f"Points du cluster {i+1}:")
        st.write(cluster)
        st.write(f"Écart-type du cluster {i+1}: {np.std(cluster, axis=0)}")
        st.write("---")

    if data.shape[1] == 2:
        colors = cm.rainbow(np.linspace(0, 1, len(clusters)))
        fig, ax = plt.subplots()
        for i, cluster in enumerate(clusters):
            cluster_data = np.array(cluster)
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f"Cluster {i+1}")
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=100, label="Centroïdes")
        ax.legend()
        st.pyplot(fig)

def predict_cluster(centroids, point):
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    cluster_id = np.argmin(distances)
    st.write(f"Le point {point} appartient au cluster {cluster_id+1}.")

def calculate_metrics(centroids, clusters, data):
    intra_cluster_distances = []
    inter_cluster_distances = []
    for i, cluster in enumerate(clusters):
        cluster_data = np.array(cluster)
        centroid = centroids[i]
        intra_cluster_distances.extend([euclidean_distance(point, centroid) for point in cluster_data])
        for j, other_cluster in enumerate(clusters):
            if j != i:
                other_centroid = centroids[j]
                inter_cluster_distances.append(euclidean_distance(centroid, other_centroid))
    intra_cluster_distance_mean = np.mean(intra_cluster_distances)
    inter_cluster_distance_mean = np.mean(inter_cluster_distances)
    st.write(f"Moyenne des distances intra-cluster: {intra_cluster_distance_mean}")
    st.write(f"Moyenne des distances inter-cluster: {inter_cluster_distance_mean}")

def suggest_k(data):
    intra_cluster_distances = []
    inter_cluster_distances = []
    k_values = range(2, min(10, data.shape[0]))
    for k in k_values:
        centroids, clusters = k_means(data, k)
        intra_dist, inter_dist = calculate_metrics_for_k(centroids, clusters, data)
        intra_cluster_distances.append(intra_dist)
        inter_cluster_distances.append(inter_dist)

    fig, ax = plt.subplots()
    ax.plot(k_values, intra_cluster_distances, label="Distances intra-cluster")
    ax.plot(k_values, inter_cluster_distances, label="Distances inter-cluster")
    ax.set_xlabel("Valeur de K")
    ax.set_ylabel("Distance moyenne")
    ax.legend()
    st.pyplot(fig)

    best_k = k_values[np.argmax(np.array(inter_cluster_distances) / np.array(intra_cluster_distances))]
    st.write(f"La valeur de K suggérée est {best_k}.")

def calculate_metrics_for_k(centroids, clusters, data):
    intra_cluster_distances = []
    inter_cluster_distances = []
    for i, cluster in enumerate(clusters):
        cluster_data = np.array(cluster)
        centroid = centroids[i]
        intra_cluster_distances.extend([euclidean_distance(point, centroid) for point in cluster_data])
        for j, other_cluster in enumerate(clusters):
            if j != i:
                other_centroid = centroids[j]
                inter_cluster_distances.append(euclidean_distance(centroid, other_centroid))
    intra_cluster_distance_mean = np.mean(intra_cluster_distances)
    inter_cluster_distance_mean = np.mean(inter_cluster_distances)
    return intra_cluster_distance_mean, inter_cluster_distance_mean

def main():
    st.title("K-means Clustering")

    st.write("Chargement des données:")
    data = []
    num_features = st.number_input("Entrez le nombre de caractéristiques:", min_value=1, step=1)
    for i in range(st.number_input("Entrez le nombre de points:", min_value=1, step=1)):
        point = []
        for j in range(num_features):
            point.append(st.number_input(f"Entrez la valeur de la caractéristique {j+1} pour le point {i+1}:"))
        data.append(point)
    data = np.array(data)

    k = st.number_input("Entrez le nombre de clusters (K):", min_value=2, step=1)

    centroids, clusters = k_means(data, k)

    display_results(centroids, clusters, data)

    predict_point = st.text_input("Entrez les valeurs des caractéristiques du point à prédire (séparées par des virgules):")
    if predict_point:
        predict_point = np.array([float(x) for x in predict_point.split(",")])
        predict_cluster(centroids, predict_point)

    calculate_metrics(centroids, clusters, data)

    suggest_k(data)

if __name__ == "__main__":
    main()