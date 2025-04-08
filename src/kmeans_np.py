import numpy as np
import random

class KMeansNP:
    def __init__(self, k, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def choose_next_centroid_np(self, data, centroids):
        #calcul de toutes les distances de chaque point à chaque centroid
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        #trouver la distance minimale de chaque point à n'importe quel centroid
        min_distances = np.min(distances, axis=1)
        #calcul de la somme des carrés des distances minimales
        somme_des_carres = np.sum(min_distances ** 2)
        #calcul des probabilités pour chaque point d'être le prochain centroid
        probabilites = (min_distances ** 2) / somme_des_carres
        #choix aléatoire du prochain centroid basé sur les probabilités calculées
        index_next_centroid = random.choices(range(len(data)), weights=probabilites, k=1)[0]
        return data[index_next_centroid]

    def kmeans_plusplus_initialization_np(self, data):
        index_first_centroid = random.randint(0, len(data) - 1) #de manière aléatoire
        centroids = [data[index_first_centroid]]
        for i in range(1, self.k):
            # choix du prochain centroid
            next_centroid = self.choose_next_centroid_np(data, centroids)
            centroids.append(next_centroid)
        return centroids
    
    def assign_points_to_clusters_np(self, data, centroids):
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        assignments = np.argmin(distances, axis=1)
        return assignments

    def update_centroids_np(self, data, assignments):
        new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def converge_np(self, old_centroids, new_centroids):
        # Calcul la distance euclidienne au carré entre les anciens et nouveaux centroids
        distances_squared = np.sum((old_centroids - new_centroids) ** 2, axis=1)
        # Vérifie si toutes les distances sont inférieures à la tolérance au carré
        return np.all(distances_squared < self.tolerance ** 2)

    def kmeans_np(self, data):
        centroids = self.kmeans_plusplus_initialization_np(data)
        for i in range(self.max_iters):
            previous_centroids = centroids.copy()
            assignments = self.assign_points_to_clusters_np(data, centroids)
            centroids = self.update_centroids_np(data, assignments)
            if self.converge_np(previous_centroids, centroids):
                break
        return assignments, centroids
    
    def fit(self,data): #fonction pour lancer l'algorithme
        return self.kmeans_np(data)