import math
import random
from collections import defaultdict

class KMedoids:
    def __init__(self, k, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def distance_euclidienne(self, point1, point2):
        # calcul de la distance euclidienne entre deux points donnés.
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def assign_points_to_clusters(self, data, medoids):
        # assignation de chaque point des données à un cluster basé sur le medoid le plus proche.
        assignments = []  # liste pour stocker l'indice du cluster de chaque point
        for point in data:
            min_distance = float('inf') 
            cluster_index = None  # initialisation de l'indice du cluster à None
            for i, medoid in enumerate(medoids):
                distance = self.distance_euclidienne(point, medoid)  # calcul de la distance au medoid
                if distance < min_distance:  # si la distance trouvée est inférieure à la distance minimale actuelle
                    min_distance = distance
                    cluster_index = i  # mise à jour de l'indice du cluster
            assignments.append(cluster_index)  # ajout de l'indice du cluster à la liste des assignations
        return assignments

    def update_medoids(self, data, assignments):
        # mise à jour des medoids pour chaque cluster pour minimiser la somme des distances dans le cluster.
        cluster_points = defaultdict(list)  # dictionnaire pour stocker les points de chaque cluster
        for assignment, point in zip(assignments, data):
            cluster_points[assignment].append(point)  # ajout du point à la liste correspondant à son cluster

        new_medoids = []  
        for points in cluster_points.values():
            min_distance_sum = float('inf')  # somme des distances minimale initialisée à l'infini
            medoid = points[0]  # médoid initial (peut être n'importe quel point du cluster)
            for point in points:
                distance_sum = sum(self.distance_euclidienne(point, other) for other in points)  # somme des distances de ce point à tous les autres points du cluster
                if distance_sum < min_distance_sum:  # si la somme des distances est inférieure à la somme minimale actuelle
                    min_distance_sum = distance_sum
                    medoid = point  # mise à jour du medoid pour ce cluster
            new_medoids.append(medoid)  # ajout du nouveau medoid à la liste
        return new_medoids

    def converge(self, old_medoids, new_medoids):
        # vérification de la convergence (aucun changement dans les medoids entre deux itérations).
        return old_medoids == new_medoids 

    def fit(self, data):
        medoids = random.sample(data, self.k)  # initialisation aléatoire des medoids
        for i in range(self.max_iters): 
            assignments = self.assign_points_to_clusters(data, medoids)  # assignation des points aux clusters
            new_medoids = self.update_medoids(data, assignments)  # mise à jour des medoids
            if self.converge(medoids, new_medoids):  # vérification de la convergence
                print("Convergence atteinte après {} itérations".format(i)) 
                break  # arrêt de la boucle si convergence atteinte
            medoids = new_medoids  # mise à jour des medoids pour la prochaine itération
        return assignments, medoids  
