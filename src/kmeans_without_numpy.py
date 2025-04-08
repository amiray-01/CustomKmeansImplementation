import math
import random
from collections import defaultdict

class KMeans_Without_Numpy:
    def __init__(self, k, max_iters=100, tolerance=1e-4): #tolérance utilisé dans le kmeans de sklearn
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def distance_euclidienne(self, point1, point2):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def choose_next_centroid(self, data, centroids):
        distances = []
        for point in data:
            min_distance = float('inf')
            for centroid in centroids:
                distance = self.distance_euclidienne(point, centroid)
                if distance < min_distance:
                    min_distance = distance
            distances.append(min_distance)

        somme_des_carres = sum(distance**2 for distance in distances) #calcul de la somme des carrés des distances
        probabilites = [distance**2 / somme_des_carres for distance in distances] #calcul des probabilités

        index_next_centroid = random.choices(range(len(data)), weights=probabilites, k=1)[0] # on effectue le choix du prochain centroid en fonction des probabilité
        return data[index_next_centroid]

    def kmeans_plusplus_initialization(self, data):
        index_first_centroid = random.randint(0, len(data) - 1) # de manière aléatoire
        centroids = [data[index_first_centroid]]
        for i in range(1, self.k):
            # choix du prochain centroid
            next_centroid = self.choose_next_centroid(data, centroids)
            centroids.append(next_centroid)
        return centroids

    def assign_points_to_clusters(self, data, centroids):
        assignments = []        # stock l'indice d'assignement du centroid
        for point in data:
            min_distance = float('inf')
            cluster_index = None
            for i, centroid in enumerate(centroids):
                distance = self.distance_euclidienne(point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    cluster_index = i
            assignments.append(cluster_index)
        return assignments

    def update_centroids(self, data, assignments):
        cluster_points = defaultdict(list)  # initialise automatiquement toute nouvelle clé avec une liste vide.
        for assignment, point in zip(assignments, data):    # regroupe les points par cluster
            cluster_points[assignment].append(point)

        new_centroids = []
        for i in range(self.k):
            points = cluster_points[i]
            if points:  # si le cluster contient des points
                new_centroid = [sum(dim)/len(dim) for dim in zip(*points)]
            else:   # si k est très grand ou si la distribution des points est très inégale
                new_centroid = random.choice(data)
            new_centroids.append(new_centroid)
        return new_centroids

    def converge(self, old_centroids, new_centroids):
        for old, new in zip(old_centroids, new_centroids): #on recupére les anciens et nouveaux centroïdes
            distance = 0
            for o, n in zip(old, new):
                distance += (o - n) ** 2 #on calcule la distance entre les anciens et nouveaux centroïdes

            if distance > self.tolerance ** 2: #si la distance est supérieur à la tolérance
                return False
        return True

    def algo_without_numpy(self, data):
        centroids = self.kmeans_plusplus_initialization(data)   #initialisation des centroïdes
        for i in range(self.max_iters):
            assignments = self.assign_points_to_clusters(data, centroids)   #assignation des points aux clusters
            new_centroids = self.update_centroids(data, assignments)    #mise à jour des centroïdes
            if self.converge(centroids, new_centroids): #si l'algo converge (les centroïdes ne bouge plus)
                break
            centroids = new_centroids   #si non on met à jour les centroïdes et on continue
        return assignments, centroids
    
    def fit(self,data): #fonction pour lancer l'algorithme
        return self.algo_without_numpy(data)